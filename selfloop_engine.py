"""
selfloop_engine.py
==================
Streamlit 앱(app_selfloop.py)이 import 하는 통합 엔진.

구성:
  - EmbeddingProvider : tok_emb pkl 있으면 사용, 없으면 결정론적 해시 폴백
  - GrowingSOM        : 자율 성장 SOM (+ INT8 양자화 저장)
  - Crawler           : "이 분야 검색해서 학습" 명령 -> 웹 수집 (네트워크 필요)
  - LLMBridge         : 답변 루프용 LLM 호출 (API 키 필요)
  - SelfLoopState     : 전체 상태 + pkl 저장/불러오기(이전 학습 누적)
  - introspect 지표   : 점유율/QE/어휘다양성/추론경로

네트워크/키가 없는 환경에서도 코퍼스 학습·계측·pkl 은 완전 작동.
크롤링/LLM 은 로컬에서 키 넣으면 작동.
"""

from __future__ import annotations
import os
import re
import time
import pickle
import hashlib
import numpy as np


# ======================================================================
# 임베딩
# ======================================================================
class EmbeddingProvider:
    """
    tok_emb pkl 형식(권장): {"word2idx": {...}, "tok_emb": np.ndarray(V, dim)}
    없으면 어절 해시 임베딩으로 폴백 (재현 가능, 네트워크 불필요).
    """
    def __init__(self, dim=64, tok_emb_path: str | None = None):
        self.dim = dim
        self.mode = "hash"
        self.word2idx = None
        self.tok_emb = None
        self._cache: dict[str, np.ndarray] = {}
        if tok_emb_path and os.path.exists(tok_emb_path):
            with open(tok_emb_path, "rb") as f:
                d = pickle.load(f)
            if "tok_emb" in d and "word2idx" in d:
                self.tok_emb = np.asarray(d["tok_emb"], dtype=np.float64)
                self.word2idx = d["word2idx"]
                self.dim = self.tok_emb.shape[1]
                self.mode = "tok_emb"

    def _word_vec(self, w: str) -> np.ndarray:
        if self.mode == "tok_emb":
            idx = self.word2idx.get(w)
            if idx is not None:
                return self.tok_emb[idx]
            return np.zeros(self.dim)
        # 해시 폴백
        if w not in self._cache:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            r = np.random.default_rng(h % (2**32))
            self._cache[w] = r.normal(size=self.dim)
        return self._cache[w]

    def encode(self, sentence: str) -> np.ndarray:
        words = sentence.split()
        if not words:
            return np.zeros(self.dim)
        vs = [self._word_vec(w) for w in words]
        return np.mean(vs, axis=0)

    def encode_many(self, sentences) -> np.ndarray:
        X = np.stack([self.encode(s) for s in sentences])
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        return X


# ======================================================================
# INT8 양자화
# ======================================================================
class Quantizer:
    def __init__(self, W):
        self.absmax = np.abs(W).max(axis=0) + 1e-8
        self.scale = self.absmax / 127.0

    def q(self, W):
        return np.clip(np.round(W / self.scale), -127, 127).astype(np.int8)

    def dq(self, Wq):
        return Wq.astype(np.float64) * self.scale


# ======================================================================
# Growing SOM
# ======================================================================
class GrowingSOM:
    def __init__(self, dim, init_nodes=16, grow_threshold=2.0, seed=0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.W = rng.normal(scale=0.3, size=(init_nodes, dim))
        self.coords = rng.normal(scale=1.0, size=(init_nodes, 2))
        self.err = np.zeros(init_nodes)
        self.grow_threshold = grow_threshold
        self.round = 0

    @property
    def n(self):
        return self.W.shape[0]

    def bmu(self, x):
        d2 = np.einsum("nd,nd->n", self.W - x, self.W - x)
        i = int(np.argmin(d2))
        return i, float(np.sqrt(d2[i]))

    def train_step(self, X, lr, radius):
        for x in X:
            i, dist = self.bmu(x)
            self.err[i] += dist
            cd2 = np.einsum("nd,nd->n", self.coords - self.coords[i],
                            self.coords - self.coords[i])
            h = np.exp(-cd2 / (2 * radius ** 2))
            self.W += (lr * h)[:, None] * (x - self.W)

    def grow(self, max_add=4):
        added = 0
        order = np.argsort(self.err)[::-1]
        for i in order[:max_add]:
            if self.err[i] < self.grow_threshold:
                break
            new_w = self.W[i] + np.random.default_rng(self.n).normal(scale=0.1, size=self.dim)
            new_c = self.coords[i] + np.random.default_rng(self.n + 1).normal(scale=0.3, size=2)
            self.W = np.vstack([self.W, new_w])
            self.coords = np.vstack([self.coords, new_c])
            self.err = np.append(self.err, 0.0)
            self.err[i] *= 0.5
            added += 1
        return added

    # ---- 추론경로 (의미 가중 최단경로) ----
    def _knn_graph(self, k=6):
        D = np.linalg.norm(self.coords[:, None] - self.coords[None, :], axis=2)
        nbr = np.argsort(D, axis=1)[:, 1:k+1]
        return nbr

    def semantic_path(self, src, dst, k=6):
        import heapq
        if src == dst:
            return [src], 0.0, 0
        nbr = self._knn_graph(k)
        N = self.n
        dist = [float("inf")] * N
        prev = [-1] * N
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == dst:
                break
            for v in nbr[u]:
                w = np.linalg.norm(self.W[u] - self.W[v])
                nd = d + w
                if nd < dist[int(v)]:
                    dist[int(v)] = nd
                    prev[int(v)] = u
                    heapq.heappush(pq, (nd, int(v)))
        if not np.isfinite(dist[dst]):
            return [src, dst], float("inf"), -1
        path, cur = [], dst
        while cur != -1:
            path.append(cur); cur = prev[cur]
        path.reverse()
        return path, dist[dst], len(path) - 1


# ======================================================================
# 계측
# ======================================================================
def measure(gsom: GrowingSOM, X, tokens):
    bmus, qes, topo_fail = [], [], 0
    for x in X:
        d2 = np.einsum("nd,nd->n", gsom.W - x, gsom.W - x)
        order = np.argsort(d2)[:2]
        bmus.append(int(order[0]))
        qes.append(float(np.sqrt(d2[order[0]])))
        # 좌표상 1·2등 인접 여부
        c1, c2 = gsom.coords[order[0]], gsom.coords[order[1]]
        if np.linalg.norm(c1 - c2) > 1.5:
            topo_fail += 1
    occ = len(set(bmus))
    all_tok = [t for seq in tokens for t in seq]
    vocab = len(set(all_tok)) / len(all_tok) if all_tok else 0.0
    return {
        "nodes": gsom.n,
        "occupancy": occ,
        "occ_ratio": occ / gsom.n,
        "mean_qe": float(np.mean(qes)) if qes else 0.0,
        "topo_error": topo_fail / len(X) if len(X) else 0.0,
        "vocab_div": vocab,
    }


def collapse_warning(history, window=3):
    if len(history) < window + 1:
        return None
    rec = history[-(window+1):]
    def down(key):
        s = [h[key] for h in rec]
        return all(b <= a for a, b in zip(s[:-1], s[1:])) and s[0] > s[-1]
    if down("occupancy") and down("vocab_div") and down("mean_qe"):
        return "붕괴 경보: 점유율·어휘·QE 동반 감소 (자기출력 메아리방)"
    if down("occupancy") and rec[-1]["occ_ratio"] < 0.1:
        return "붕괴 경보: 점유율 10%↓ 한 점 수렴"
    return None


# ======================================================================
# 크롤러 (네트워크 필요 — 로컬 실행 시 작동)
# ======================================================================
def crawl_topic(topic: str, max_pages=5):
    """
    '이 분야 검색해서 학습' 명령 처리.
    DuckDuckGo HTML 검색 -> 본문 문장 추출. 네트워크 없으면 예외.
    """
    import urllib.request, urllib.parse
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.skip = False
            self.parts = []
        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.skip = True
        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self.skip = False
        def handle_data(self, data):
            if not self.skip:
                t = data.strip()
                if t:
                    self.parts.append(t)

    q = urllib.parse.quote(topic)
    url = f"https://html.duckduckgo.com/html/?q={q}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", "ignore")
    links = re.findall(r'href="(https?://[^"]+)"', html)[:max_pages]

    sentences = []
    for link in links:
        try:
            r = urllib.request.Request(link, headers={"User-Agent": "Mozilla/5.0"})
            page = urllib.request.urlopen(r, timeout=10).read().decode("utf-8", "ignore")
            ex = TextExtractor(); ex.feed(page)
            text = " ".join(ex.parts)
            for s in re.split(r'(?<=[.!?。])\s+|\n', text):
                s = s.strip()
                if 10 <= len(s) <= 200:
                    sentences.append(s)
        except Exception:
            continue
    # 중복 제거
    seen, out = set(), []
    for s in sentences:
        if s not in seen:
            seen.add(s); out.append(s)
    return out[:300]


# ======================================================================
# LLM 브리지 (답변 루프 — API 키 필요)
# ======================================================================
def llm_answer(
    question: str,
    context_sentences,
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=400,
    api_key: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
):
    """
    학습된 SOM 코퍼스에서 뽑은 맥락으로 OpenAI-compatible LLM 답변 생성.

    - api_key가 전달되면 우선 사용하고, 없으면 OPENAI_API_KEY 환경변수를 사용합니다.
    - base_url을 전달하면 OpenAI 호환 서버(Ollama proxy, vLLM, LM Studio 등)에도 연결할 수 있습니다.
      예: https://api.openai.com/v1 또는 http://localhost:1234/v1
    """
    try:
        from openai import OpenAI
    except Exception:
        return "[openai 미설치] pip install openai 후 사용"

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return "[API KEY 미설정] 사이드바에 API Key를 입력하거나 OPENAI_API_KEY 환경변수를 설정하세요."

    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    client = OpenAI(**kwargs)

    ctx = "\n".join(f"- {s}" for s in context_sentences[:15])
    system = system_prompt or (
        "너는 학습된 코퍼스 범위 안에서만 답한다. "
        "맥락에 없는 내용은 추측하지 말고 모른다고 말한다. "
        "답변은 사용자가 이해하기 쉽게 하되, 학습된 맥락과 충돌하지 않게 작성한다."
    )
    user = f"[학습된 맥락]\n{ctx}\n\n[질문]\n{question}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[LLM 호출 실패] {type(e).__name__}: {e}"


# ======================================================================
# 전체 상태 + pkl 누적 저장/불러오기
# ======================================================================
class SelfLoopState:
    def __init__(self, dim=64):
        self.dim = dim
        self.gsom = GrowingSOM(dim=dim)
        self.corpus: list[str] = []       # 학습된 모든 문장(코퍼스)
        self.history: list[dict] = []     # 라운드별 계측
        self.created = time.time()

    def save(self, path):
        # SOM 가중치는 INT8 양자화해서 저장 (용량 보완)
        quant = Quantizer(self.gsom.W)
        blob = {
            "dim": self.dim,
            "W_int8": quant.q(self.gsom.W),
            "W_scale": quant.scale,
            "coords": self.gsom.coords,
            "err": self.gsom.err,
            "round": self.gsom.round,
            "corpus": self.corpus,
            "history": self.history,
            "created": self.created,
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            b = pickle.load(f)
        st = cls(dim=b["dim"])
        # 역양자화로 SOM 복원
        st.gsom.W = b["W_int8"].astype(np.float64) * b["W_scale"]
        st.gsom.coords = b["coords"]
        st.gsom.err = b["err"]
        st.gsom.round = b.get("round", 0)
        st.corpus = b["corpus"]
        st.history = b["history"]
        st.created = b.get("created", time.time())
        return st
