"""
app_selfloop.py
===============
자율 학습 루프 + 답변 루프 통합 Streamlit 앱.

실행:
    pip install streamlit numpy openai
    streamlit run app_selfloop.py

페이지
  1) 학습 루프  : 코퍼스 업로드 / "분야 검색 학습"(크롤링) / 라운드 옵션 / 계측 / pkl 저장
  2) 답변 루프  : pkl 불러오기 / 질문 / SOM 가드레일 / LLM 호출 / 추론경로

tok_emb 연결: 사이드바에서 tok_emb pkl 업로드 (없으면 해시 임베딩 폴백)
이전 학습 누적: pkl 불러오기 -> 같은 상태에 계속 학습
"""

import io
import os
import time
import pickle
import numpy as np
import streamlit as st

from selfloop_engine import (
    EmbeddingProvider, SelfLoopState, measure, collapse_warning,
    crawl_topic, llm_answer,
)

st.set_page_config(page_title="SelfLoop SOM", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background:#0d1117; color:#c9d1d9; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.02em; color:#e6edf3; }
.mono, code, pre { font-family:'JetBrains Mono',monospace !important; }
.metric-card{ background:#161b22; border:1px solid #21262d; border-left:3px solid #2dd4bf;
  padding:14px 16px; border-radius:6px; }
.metric-val{ font-family:'JetBrains Mono',monospace; font-size:1.7rem; color:#2dd4bf; font-weight:600;}
.metric-lab{ font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#8b949e;}
.warn{ background:#3d1b1b; border-left:3px solid #f85149; padding:10px 14px; border-radius:6px;
  color:#ffa198; font-family:'JetBrains Mono',monospace; font-size:.85rem;}
.ok{ background:#12261e; border-left:3px solid #2dd4bf; padding:10px 14px; border-radius:6px;
  color:#7ee2cf; font-family:'JetBrains Mono',monospace; font-size:.85rem;}
.stButton>button{ background:#2dd4bf; color:#0d1117; border:none; font-weight:600;
  font-family:'Space Grotesk',sans-serif; border-radius:6px;}
.stButton>button:hover{ background:#5eead4; color:#0d1117;}
.pathbox{ background:#161b22; border:1px solid #21262d; padding:12px; border-radius:6px;
  font-family:'JetBrains Mono',monospace; font-size:.8rem; color:#8b949e;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- state
if "state" not in st.session_state:
    st.session_state.state = SelfLoopState(dim=64)
if "emb" not in st.session_state:
    st.session_state.emb = EmbeddingProvider(dim=64)

def metric(col, label, value):
    col.markdown(f'<div class="metric-card"><div class="metric-val">{value}</div>'
                 f'<div class="metric-lab">{label}</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙ 설정")
    up = st.file_uploader("tok_emb pkl (선택)", type=["pkl"], key="tokemb")
    if up is not None:
        tmp = f"/tmp/{up.name}"
        with open(tmp, "wb") as f:
            f.write(up.getbuffer())
        st.session_state.emb = EmbeddingProvider(dim=64, tok_emb_path=tmp)
        st.session_state.state.dim = st.session_state.emb.dim
    mode = st.session_state.emb.mode
    st.markdown(f'<div class="{"ok" if mode=="tok_emb" else "warn"}">임베딩: '
                f'{"TinyTransformer tok_emb" if mode=="tok_emb" else "해시 폴백(tok_emb 없음)"}'
                f'</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔌 GPT / LLM API")
    api_key_input = st.text_input(
        "API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="OpenAI 또는 OpenAI 호환 서버의 API Key. 입력값은 세션에서만 사용됩니다."
    )
    base_url_input = st.text_input(
        "Base URL",
        value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI 기본값: https://api.openai.com/v1 / LM Studio 예: http://localhost:1234/v1"
    )
    max_tokens_input = st.number_input("max_tokens", min_value=128, max_value=4096, value=700, step=128)
    system_prompt_input = st.text_area(
        "System Prompt",
        value=("너는 학습된 코퍼스 범위 안에서만 답한다. "
               "맥락에 없는 내용은 추측하지 말고 모른다고 말한다. "
               "학습된 의미지도와 도메인 정체성을 유지한다."),
        height=110
    )

    st.divider()
    stt = st.session_state.state
    st.markdown(f"**코퍼스** `{len(stt.corpus)}` 문장")
    st.markdown(f"**노드** `{stt.gsom.n}`")
    st.markdown(f"**라운드 누적** `{stt.gsom.round}`")

    st.divider()
    st.markdown("### 💾 세션")
    if st.button("현재 상태 pkl 저장"):
        buf = io.BytesIO()
        stt.save("/tmp/_save.pkl")
        with open("/tmp/_save.pkl", "rb") as f:
            st.download_button("⬇ selfloop_state.pkl 내려받기", f.read(),
                               file_name="selfloop_state.pkl")
    rl = st.file_uploader("pkl 불러오기(이전 학습 누적)", type=["pkl"], key="loadpkl")
    if rl is not None and st.button("불러와서 이어가기"):
        with open("/tmp/_load.pkl", "wb") as f:
            f.write(rl.getbuffer())
        st.session_state.state = SelfLoopState.load("/tmp/_load.pkl")
        st.success(f"복원: {len(st.session_state.state.corpus)}문장 / "
                   f"{st.session_state.state.gsom.n}노드")
        st.rerun()

# ---------------------------------------------------------------- header
st.markdown("# SelfLoop SOM")
st.markdown("<span class='mono' style='color:#8b949e'>자율 성장 의미지도 · 학습 루프와 답변 루프</span>",
            unsafe_allow_html=True)

page = st.tabs(["◐  학습 루프", "◑  답변 루프"])

# ======================================================================
# PAGE 1 — 학습 루프
# ======================================================================
with page[0]:
    stt = st.session_state.state
    emb = st.session_state.emb

    st.markdown("### 1 · 학습 자료")
    src = st.radio("자료 출처", ["코퍼스 직접 입력/업로드", "분야 검색해서 학습(크롤링)"],
                   horizontal=True)

    new_sentences = []
    if src == "코퍼스 직접 입력/업로드":
        c1, c2 = st.columns(2)
        txt = c1.text_area("문장 직접 입력 (줄바꿈으로 구분)", height=160,
                           placeholder="오늘 날씨가 좋아요\n밥을 먹어요\n...")
        f = c2.file_uploader("또는 .txt 코퍼스 업로드", type=["txt"])
        if txt.strip():
            new_sentences += [s.strip() for s in txt.splitlines() if s.strip()]
        if f is not None:
            content = f.getvalue().decode("utf-8", "ignore")
            new_sentences += [s.strip() for s in content.splitlines() if s.strip()]
    else:
        topic = st.text_input("학습할 분야/검색어",
                              placeholder="예: 발달장애 학생 읽기 쉬운 자료")
        npg = st.slider("크롤링 페이지 수", 1, 10, 5)
        if st.button("🔍 검색해서 수집"):
            with st.spinner(f"'{topic}' 크롤링 중..."):
                try:
                    new_sentences = crawl_topic(topic, max_pages=npg)
                    st.session_state._crawled = new_sentences
                    st.success(f"{len(new_sentences)}문장 수집")
                except Exception as e:
                    st.markdown(f'<div class="warn">크롤링 실패: {e}<br>'
                                f'(이 환경은 네트워크 차단. 로컬에서 작동)</div>',
                                unsafe_allow_html=True)
        new_sentences = st.session_state.get("_crawled", [])
        if new_sentences:
            st.caption(f"수집된 문장 미리보기 ({len(new_sentences)}개)")
            st.code("\n".join(new_sentences[:8]), language=None)

    st.markdown("### 2 · 라운드 옵션")
    c1, c2, c3 = st.columns(3)
    mode_r = c1.selectbox("종료 기준", ["라운드 수", "1분", "5분", "10분"])
    n_rounds = c2.number_input("라운드 수", 1, 200, 10, disabled=(mode_r != "라운드 수"))
    grow_on = c3.checkbox("GSOM 자율 성장", value=True)

    if st.button("▶ 학습 시작", type="primary"):
        if new_sentences:
            stt.corpus += new_sentences
        if not stt.corpus:
            st.markdown('<div class="warn">학습할 코퍼스가 없습니다.</div>',
                        unsafe_allow_html=True)
        else:
            X = emb.encode_many(stt.corpus)
            toks = [s.split() for s in stt.corpus]
            time_limit = {"1분":60,"5분":300,"10분":600}.get(mode_r)
            t_start = time.time()
            prog = st.progress(0.0)
            live = st.empty()
            r = 0
            while True:
                r += 1
                stt.gsom.round += 1
                lr = 0.4 * (0.9 ** stt.gsom.round)
                rad = max(0.5, 2.0 * (0.85 ** stt.gsom.round))
                stt.gsom.train_step(X, lr, rad)
                grew = stt.gsom.grow() if grow_on else 0
                m = measure(stt.gsom, X, toks)
                m["round"] = stt.gsom.round; m["grew"] = grew
                stt.history.append(m)
                w = collapse_warning(stt.history)
                live.markdown(
                    f"<div class='{'warn' if w else 'ok'}'>r{stt.gsom.round} · "
                    f"nodes={m['nodes']} occ={m['occupancy']} "
                    f"qe={m['mean_qe']:.2f} vocab={m['vocab_div']:.2f}"
                    f"{' · '+w if w else ''}</div>", unsafe_allow_html=True)
                if time_limit:
                    prog.progress(min(1.0, (time.time()-t_start)/time_limit))
                    if time.time()-t_start >= time_limit: break
                else:
                    prog.progress(min(1.0, r/n_rounds))
                    if r >= n_rounds: break
            st.success(f"학습 완료 · 총 {r}라운드 · 노드 {stt.gsom.n}")

    # ---- 계측 차트 ----
    if stt.history:
        st.markdown("### 3 · 계측")
        h = stt.history
        rounds = [x["round"] for x in h]
        c1,c2,c3,c4 = st.columns(4)
        last = h[-1]
        metric(c1,"NODES", last["nodes"])
        metric(c2,"OCCUPANCY", last["occupancy"])
        metric(c3,"MEAN QE", f"{last['mean_qe']:.2f}")
        metric(c4,"VOCAB DIV", f"{last['vocab_div']:.2f}")

        import pandas as pd
        df = pd.DataFrame(h).set_index("round")
        cc1, cc2 = st.columns(2)
        cc1.caption("점유율 / 노드")
        cc1.line_chart(df[["occupancy","nodes"]])
        cc2.caption("평균 QE / 어휘다양성")
        cc2.line_chart(df[["mean_qe","vocab_div"]])

        w = collapse_warning(h)
        if w:
            st.markdown(f'<div class="warn">⚠ {w} — 인간 피드백 필요 시점</div>',
                        unsafe_allow_html=True)

# ======================================================================
# PAGE 2 — 답변 루프
# ======================================================================
with page[1]:
    stt = st.session_state.state
    emb = st.session_state.emb

    if not stt.corpus:
        st.markdown('<div class="warn">먼저 학습 루프에서 코퍼스를 학습하거나 '
                    '사이드바에서 pkl을 불러오세요.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"### 학습된 지도에 질문하기")
        st.caption(f"코퍼스 {len(stt.corpus)}문장 · {stt.gsom.n}노드 위에서 답변")

        c1, c2, c3 = st.columns(3)
        model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "gpt-4-turbo", "직접 입력"]
        model_pick = c1.selectbox("LLM 모델", model_options)
        model = c1.text_input("모델 직접 입력", value="gpt-4o-mini") if model_pick == "직접 입력" else model_pick
        temp = c2.slider("temperature", 0.0, 1.0, 0.3, 0.1)
        topk = c3.slider("맥락 문장 수", 3, 20, 10)
        save_qa = st.checkbox("질문·답변을 코퍼스에 저장(자기성찰 누적)", value=True)

        q = st.text_input("질문", placeholder="학습한 내용을 바탕으로 물어보세요")

        if st.button("▶ 답변 생성", type="primary") and q.strip():
            qv = emb.encode(q)
            # SOM 가드레일: 질문이 학습 분포 안인가 (BMU 거리)
            qbmu, qdist = stt.gsom.bmu(qv)
            # 코퍼스에서 질문 BMU에 가까운 문장들 = 맥락
            Xc = emb.encode_many(stt.corpus)
            sims = [-np.linalg.norm(emb.encode(s) - qv) for s in stt.corpus]
            order = np.argsort(sims)[::-1][:topk]
            context = [stt.corpus[i] for i in order]

            # 가드레일 판정
            qe_vals = [h["mean_qe"] for h in stt.history] or [qdist]
            band = float(np.mean(qe_vals) + 2*np.std(qe_vals)) if len(qe_vals)>1 else qdist*1.5
            in_domain = qdist <= band

            cL, cR = st.columns([3,2])
            with cL:
                if in_domain:
                    st.markdown(f'<div class="ok">✓ 도메인 내 질문 '
                                f'(BMU거리 {qdist:.2f} ≤ 경계 {band:.2f})</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warn">⚠ 도메인 이탈 의심 '
                                f'(BMU거리 {qdist:.2f} > 경계 {band:.2f}) — '
                                f'학습 범위 밖일 수 있음</div>', unsafe_allow_html=True)

                with st.spinner("LLM 호출 중..."):
                    ans = llm_answer(
                        q,
                        context,
                        model=model,
                        temperature=temp,
                        max_tokens=int(max_tokens_input),
                        api_key=api_key_input.strip() or None,
                        base_url=base_url_input.strip() or None,
                        system_prompt=system_prompt_input.strip() or None,
                    )
                st.markdown("#### 답변")
                st.write(ans)

                if save_qa and ans and not ans.startswith("["):
                    stt.corpus.append(f"질문: {q}")
                    stt.corpus.append(f"답변: {ans}")
                    st.info("질문·답변을 코퍼스에 저장했습니다. 다음 학습 라운드에서 의미지도에 반영됩니다.")

            with cR:
                st.markdown("#### 사용된 맥락")
                st.markdown('<div class="pathbox">'+"<br>".join(
                    f"· {c}" for c in context[:8])+'</div>',
                    unsafe_allow_html=True)
                # 추론경로: 질문 BMU -> 답변에 가장 가까운 코퍼스 BMU
                abmu, _ = stt.gsom.bmu(emb.encode(context[0]))
                path, cost, hops = stt.gsom.semantic_path(qbmu, abmu)
                st.markdown("#### 추론경로 (논리의 발자국)")
                st.markdown(f'<div class="pathbox">질문격자 #{qbmu} → 답변격자 #{abmu}<br>'
                            f'의미 도정거리 = <b style="color:#2dd4bf">{cost:.2f}</b><br>'
                            f'거친 노드 = <b style="color:#2dd4bf">{hops}</b><br>'
                            f'경로: {" → ".join("#"+str(p) for p in path[:10])}'
                            f'{" ..." if len(path)>10 else ""}</div>',
                            unsafe_allow_html=True)

        st.divider()
        st.caption("팁: 답변 저장을 켜면 Q·A가 코퍼스에 누적됩니다. 이후 학습 루프를 다시 돌리면 SOM 지도와 도메인 경계에 반영됩니다.")
