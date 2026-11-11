"""DocChat: Streamlit RAG chat UI.  Run:  python -m streamlit run app.py"""
import html
import os
import time
from pathlib import Path

import streamlit as st

esc = html.escape

from rag import INDEX_DIR, answer_stream, get_llm, load_store

st.set_page_config(page_title="DocChat | Document Q&A", page_icon="📄", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500&display=swap');
:root { --bg:#e7e6de; --ink:#101010; --lime:#c9ea6d; --forest:#17361f; --paper:#f3f2ea; --line:#cfcec4; --mid:#6f6f68;
  --head:'Bricolage Grotesque','Helvetica Neue',Arial,sans-serif; --serif:'Instrument Serif',Georgia,serif;
  --sans:'Inter','Helvetica Neue',Arial,sans-serif; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:var(--bg) !important; font-family:var(--sans); color:var(--ink); }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { max-width:none !important; padding:1.4rem clamp(1rem,3vw,3.4rem) 9rem !important; }
em { font-family:var(--serif); font-style:italic; font-weight:400; }
[data-testid="stMarkdownContainer"] a { text-decoration:none !important; color:inherit !important; }

/* nav */
.nav { display:flex; justify-content:space-between; align-items:center; padding:.2rem 0 1.2rem; }
.logo { font-family:var(--head); font-weight:800; font-size:1.55rem; letter-spacing:-.045em; }
.navr { display:flex; gap:.45rem; }
.np { border:1px solid var(--ink); border-radius:999px; padding:.4rem 1.05rem; font-size:.72rem; transition:all .15s; }
[data-testid="stMarkdownContainer"] a.np.on, [data-testid="stMarkdownContainer"] a.np:hover { background:var(--ink); color:#fff !important; }

/* hero */
.hero { display:grid; grid-template-columns:1fr 1.25fr; gap:1rem; min-height:560px; }
.lcol { display:flex; flex-direction:column; gap:1rem; }
.gcard { background:var(--lime); border-radius:46px; padding:clamp(1.6rem,3vw,3rem); flex:1;
  -webkit-mask:radial-gradient(circle at 100% 100%, transparent 0 78px, #000 79px);
  mask:radial-gradient(circle at 100% 100%, transparent 0 78px, #000 79px); }
.ht { font-family:var(--head); font-weight:800; font-size:clamp(2.2rem,3.9vw,4.3rem); line-height:1.06; letter-spacing:-.045em; margin:0 0 1.3rem; }
.hl { background:#fff; border-radius:12px; padding:0 .2em; -webkit-box-decoration-break:clone; box-decoration-break:clone; }
.hp { font-size:.9rem; line-height:1.6; max-width:27rem; margin:0 0 1.5rem; color:#2c3517; }
.gbtn { display:inline-block; background:#fff; border-radius:999px; padding:.75rem 1.5rem; font-size:.8rem; font-weight:500; }
.dcard { border:2px dashed #a9a89d; border-radius:38px; padding:1.5rem 1.7rem; }
.dn { font-family:var(--head); font-weight:800; font-size:clamp(2.4rem,4vw,3.8rem); letter-spacing:-.05em; line-height:1; }
.dl { font-family:var(--head); font-weight:700; font-size:1.2rem; margin:.25rem 0 1rem; letter-spacing:-.02em; }
.dav { display:flex; }
.dav span { width:38px; height:38px; border-radius:50%; border:3px solid var(--bg); margin-left:-8px; display:flex; align-items:center;
  justify-content:center; font-size:.7rem; font-weight:700; }
.dav span:first-child { margin-left:0; }
.dfoot { margin-top:1rem; font-size:.74rem; color:var(--mid); display:flex; gap:.35rem; flex-wrap:wrap; }
.dfoot b { color:var(--ink); font-weight:500; }

.rcol { position:relative; background:var(--forest); border-radius:46px; overflow:hidden; min-height:560px;
  background-image:radial-gradient(70% 60% at 80% 8%, rgba(201,234,109,.2), transparent 60%), radial-gradient(60% 50% at 8% 92%, rgba(255,255,255,.08), transparent 60%); }
.sheet { position:absolute; width:34%; aspect-ratio:3/4; background:#fff; border-radius:16px; padding:1rem; box-shadow:0 18px 36px rgba(0,0,0,.3); overflow:hidden; }
.sheet .stl { font-family:var(--head); font-weight:700; font-size:clamp(.7rem,1vw,.95rem); letter-spacing:-.02em; line-height:1.15; margin-bottom:.7rem; }
.sheet .ln { height:64%; background:repeating-linear-gradient(to bottom,#dedcd2 0 2px,transparent 2px 11px); }
.sheet .ex { font-size:clamp(.5rem,.72vw,.7rem); line-height:1.55; color:#66665e; overflow:hidden; height:72%;
  -webkit-mask-image:linear-gradient(#000 62%,transparent); mask-image:linear-gradient(#000 62%,transparent); }
.s0 { left:7%; top:9%; transform:rotate(-7deg); } .s1 { left:33%; top:20%; transform:rotate(4deg); } .s2 { left:12%; top:48%; transform:rotate(3deg); }
.k { position:absolute; display:flex; align-items:center; justify-content:center; text-align:center; font-family:var(--head);
  font-weight:800; letter-spacing:-.03em; line-height:1.05; }
.k1 { right:9%; top:15%; width:clamp(96px,10vw,150px); aspect-ratio:1; background:var(--lime); border-radius:28px; transform:rotate(-8deg);
  flex-direction:column; gap:.4rem; font-size:clamp(.9rem,1.4vw,1.4rem); }
.k2 { right:5%; top:44%; width:clamp(78px,7.5vw,112px); aspect-ratio:1; background:#fff; border-radius:50%; transform:rotate(-14deg);
  font-size:clamp(.62rem,.9vw,.85rem); padding:.6rem; }
.k3 { left:6%; bottom:7%; width:clamp(100px,10.5vw,150px); aspect-ratio:1; background:#0c2113; color:var(--lime); border-radius:50%;
  transform:rotate(-16deg); font-size:clamp(.8rem,1.2vw,1.2rem); padding:.8rem; }
.k4 { left:36%; bottom:11%; background:#fff; border-radius:14px; padding:.6rem 1.1rem; transform:rotate(-8deg); font-size:clamp(.8rem,1.2vw,1.15rem); }
.k5 { right:6%; bottom:6%; width:clamp(96px,9.5vw,138px); aspect-ratio:1; background:var(--lime);
  border-radius:30% 70% 62% 38% / 40% 38% 62% 60%; transform:rotate(10deg); font-size:clamp(.9rem,1.3vw,1.3rem); }
@media (max-width:900px){ .hero{grid-template-columns:1fr;} .rcol{min-height:440px;} .gcard{-webkit-mask:none; mask:none;} }

/* section headers + timeline */
.sechead { display:flex; align-items:center; gap:1.4rem; margin:3.4rem 0 1.4rem; }
.stitle { font-family:var(--head); font-weight:800; font-size:clamp(1.8rem,3.2vw,2.9rem); letter-spacing:-.045em; line-height:1; white-space:nowrap; }
.tl { flex:1; display:flex; align-items:center; gap:.8rem; font-size:.72rem; color:var(--mid); }
.tl i { flex:1; height:1px; background:#b9b8ad; position:relative; }
.tl i b { position:absolute; left:24%; top:-5px; width:11px; height:11px; border-radius:50%; background:var(--ink); }

/* widgets */
[data-testid="stWidgetLabel"] p { font-size:.7rem !important; color:var(--mid) !important; font-family:var(--sans); }
.stButton > button { background:transparent !important; color:var(--ink) !important; border:1px solid var(--ink) !important; border-radius:999px !important;
  font-family:var(--sans); font-weight:500; padding:.6rem 1.2rem; text-align:left; transition:all .15s; }
.stButton > button p { color:inherit !important; font-size:.82rem; }
.stButton > button:hover { background:var(--lime) !important; color:var(--ink) !important; border-color:var(--ink) !important; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background:transparent !important; }
[data-testid="stChatInput"] { max-width:960px; margin:0 auto; background:#fff !important; border:1px solid var(--ink) !important;
  border-radius:999px !important; box-shadow:0 10px 30px rgba(0,0,0,.08); }
[data-testid="stChatInput"] div { border:none !important; box-shadow:none !important; outline:none !important; background:transparent !important; }
[data-testid="stChatInput"] textarea { background:transparent !important; font-family:var(--sans) !important; box-shadow:none !important; outline:none !important; }
[data-testid="stChatInput"]:focus-within { border-color:var(--ink) !important; box-shadow:0 10px 30px rgba(0,0,0,.12) !important; }
[data-testid="stChatInputSubmitButton"] { background:var(--ink) !important; color:#fff !important; border-radius:50% !important; }

/* chat */
.thinking { display:flex; align-items:center; gap:.45rem; color:var(--mid); font-size:.88rem; padding:.2rem 0 .4rem; }
.thinking span { width:6px; height:6px; border-radius:50%; background:var(--ink); animation:blink 1s infinite both; }
.thinking span:nth-child(2) { animation-delay:.18s; } .thinking span:nth-child(3) { animation-delay:.36s; }
.thinking em { font-family:var(--sans); font-style:normal; margin-left:.2rem; }
@keyframes blink { 0%,80%,100% { opacity:.15; } 40% { opacity:1; } }
.try { font-family:var(--head); font-weight:800; font-size:clamp(1.4rem,2.4vw,2.1rem); letter-spacing:-.04em; margin:.4rem 0 .9rem; }
[class*="st-key-u_"] { background:var(--ink); border-radius:26px 26px 6px 26px; padding:.9rem 1.4rem .9rem; width:fit-content; max-width:72%; margin-left:auto; }
[class*="st-key-u_"] * { color:#fff !important; }
[class*="st-key-u_"] p { font-size:.95rem; }
[class*="st-key-a_"] { background:#fff; border-radius:30px 30px 30px 6px; padding:1.3rem 1.6rem 1.5rem; max-width:1100px; margin:.4rem 0 1.4rem; }
.atags { display:flex; gap:.4rem; margin-bottom:.7rem; }
.tag { background:var(--lime); border-radius:999px; padding:.22rem .7rem; font-size:.66rem; font-weight:500; }
.tag.t2 { background:var(--paper); color:var(--mid); }
.srclab { font-family:var(--head); font-weight:700; font-size:1rem; letter-spacing:-.02em; margin:1.3rem 0 .2rem; }
.srcrow { display:flex; gap:.8rem; align-items:flex-start; flex-wrap:wrap; margin-top:.7rem; }
.src { flex:1 1 170px; min-width:0; }
.snip { background:var(--paper); border-radius:14px; padding:.85rem .9rem; font-size:.7rem; line-height:1.55; color:#5d5d56; overflow:hidden; }
.src:nth-child(odd) .snip { height:172px; } .src:nth-child(even) .snip { height:122px; }
.sline { display:flex; gap:.55rem; align-items:flex-start; margin-top:.55rem; }
.sn { font-family:var(--head); font-weight:500; font-size:2.2rem; letter-spacing:-.05em; line-height:.95; }
.sm b { display:block; font-size:.74rem; font-weight:500; margin-top:.15rem; }
.sm span { font-size:.68rem; color:var(--mid); }

/* library */
.lib { display:flex; gap:.9rem; align-items:flex-start; flex-wrap:wrap; }
.doc { flex:1 1 150px; min-width:0; }
.cov { border-radius:16px; position:relative; overflow:hidden; padding:1rem; display:flex; flex-direction:column; gap:.7rem; }
.cov .cchip { align-self:flex-start; background:rgba(255,255,255,.7); border-radius:999px; padding:.2rem .65rem; font-size:.6rem; font-weight:500; }
.cov .cex { flex:1; overflow:hidden; font-size:.72rem; line-height:1.6; color:rgba(0,0,0,.58);
  -webkit-mask-image:linear-gradient(#000 55%,transparent); mask-image:linear-gradient(#000 55%,transparent); }
.cov.noex::after { content:""; position:absolute; inset:14px; top:44px; background:repeating-linear-gradient(to bottom,rgba(0,0,0,.14) 0 1.5px,transparent 1.5px 10px); opacity:.55; border-radius:6px; }
.doc:nth-child(odd) .cov { height:220px; } .doc:nth-child(even) .cov { height:150px; }
.bigfoot { display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; margin-top:2.6rem; flex-wrap:wrap; }
.bl { font-size:.78rem; display:flex; gap:1.3rem; flex-wrap:wrap; padding-bottom:1rem; }
.bl span { color:var(--mid); } .bl b { font-weight:500; }
.bw { font-family:var(--serif); font-style:italic; font-size:clamp(4rem,12vw,10.5rem); line-height:.88; letter-spacing:-.03em; }

/* how it works */
.how { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; }
@media (max-width:900px){ .how{grid-template-columns:1fr;} }
.hc { border-radius:36px; padding:1.8rem 1.8rem 2rem; background:#fff; }
.hc.l { background:var(--lime); } .hc.d { background:var(--forest); color:#fff; }
.hc .hn { font-family:var(--serif); font-style:italic; font-size:3rem; line-height:1; }
.hc .ht2 { font-family:var(--head); font-weight:800; font-size:1.5rem; letter-spacing:-.04em; margin:.8rem 0 .5rem; }
.hc .hd { font-size:.82rem; line-height:1.6; opacity:.8; }
.footnote { margin-top:2.4rem; font-size:.7rem; color:var(--mid); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = [
    "How many attention heads does the base Transformer use?",
    "What two tasks is BERT pre-trained on?",
    "What is the difference between RAG-Sequence and RAG-Token?",
    "What does DCGAN use instead of pooling layers?",
]
COVERS = ["#c9ea6d", "#f4efdc", "#bcd9b0", "#f3cfc0", "#d9d6f2", "#fbe38e", "#cfe3ea"]
PRETTY = {
    "bert": "BERT", "dcgan": "DCGAN", "resnet": "ResNet", "word2vec": "Word2Vec",
    "gan goodfellow": "GAN (Goodfellow)", "rag retrieval augmented generation": "RAG",
}


def pretty(stem):
    key = stem.replace("_", " ").lower()
    return PRETTY.get(key, key.title())


def excerpt(name, n=600):
    text = " ".join(str(excerpts.get(name, "")).split())
    return esc(text[:n])


if not os.path.isdir(INDEX_DIR):
    st.error("No index found. Run `python download_docs.py` then `python ingest.py` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading index and model...")
def init():
    store = load_store()
    stats, first = {}, {}
    try:  # chunks, page count and a first-page excerpt per document, read from the index itself
        for d in store.docstore._dict.values():
            name = d.metadata.get("source_name")
            page = int(d.metadata.get("page_label") or 0)
            s = stats.setdefault(name, {"chunks": 0, "pages": 0})
            s["chunks"] += 1
            s["pages"] = max(s["pages"], page)
            if name not in first or page < first[name][0]:
                first[name] = (page, d.page_content)
    except Exception:
        pass
    try:  # load the embedding model now so the first question isn't slow
        store.similarity_search("warm up", k=1)
    except Exception:
        pass
    return store, get_llm(), stats, {k: v[1] for k, v in first.items()}


store, llm, stats, excerpts = init()
doc_files = sorted(p for p in Path("docs").glob("*") if p.suffix.lower() in (".pdf", ".txt"))
n_docs, n_chunks = len(doc_files), store.index.ntotal
provider = os.getenv("LLM_PROVIDER", "gemini").title()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- nav + hero ----------
titles = [pretty(f.stem) for f in doc_files]
initials = "".join(
    f'<span style="background:{COVERS[i % len(COVERS)]}">{esc(t[0])}</span>' for i, t in enumerate(titles[:7])
)
sheets = "".join(
    f'<div class="sheet s{i}"><div class="stl">{esc(t)}</div>'
    + (f'<div class="ex">{excerpt(doc_files[i].name, 900)}</div>' if excerpts.get(doc_files[i].name) else '<div class="ln"></div>')
    + "</div>"
    for i, t in enumerate(titles[:3])
)
st.markdown(
    f"""
<div class="nav"><div class="logo">DocChat.</div>
  <div class="navr"><a class="np on" href="#chat" target="_self">Chat</a><a class="np" href="#library" target="_self">Library</a>
  <a class="np" href="#how" target="_self">How it works</a></div></div>
<div class="hero">
  <div class="lcol">
    <div class="gcard">
      <div class="ht">Get <span class="hl">grounded answers</span> from your documents.</div>
      <p class="hp">Looking for a fact buried in a paper? DocChat searches your documents, finds the passages that matter,
        and answers with the sources attached.</p>
      <a class="gbtn" href="#chat" target="_self">Get started</a>
    </div>
    <div class="dcard"><div class="dn">{n_chunks:,}+</div><div class="dl">Passages indexed</div>
      <div class="dav">{initials}</div>
      <div class="dfoot"><b>{provider}</b> · <b>FAISS</b> · <b>LangChain</b> · {n_docs} documents</div></div>
  </div>
  <div class="rcol">{sheets}
    <div class="k k1"><span>&#10003;</span>Source-grounded</div>
    <div class="k k2">{n_chunks:,} passages</div>
    <div class="k k3">No made-up answers</div>
    <div class="k k4">all-MiniLM-L6-v2</div>
    <div class="k k5">FAISS</div>
  </div>
</div>
<div id="chat"></div>
<div class="sechead"><div class="stitle">Ask <em>anything</em></div>
  <div class="tl"><span>01.</span><i><b></b></i><span>{max(n_docs, 1):02d}.</span></div></div>
""",
    unsafe_allow_html=True,
)

# ---------- chat helpers ----------
with st.container(key="ctrl"):
    c1, c2 = st.columns([3, 1])
    top_k = c1.slider("Passages to retrieve", 1, 8, 4)
    c2.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    if c2.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def sources_html(sources):
    cards = "".join(
        f'<div class="src"><div class="snip">{esc(s["text"][:260])}...</div><div class="sline"><div class="sn">{i:02d}.</div>'
        f'<div class="sm"><b>{esc(pretty(Path(str(s["name"])).stem))}</b><span>Page {s["page"]}</span></div></div></div>'
        for i, s in enumerate(sources, start=1)
    )
    return f'<div class="srclab">Sources</div><div class="srcrow">{cards}</div>'


def tags_html(elapsed=None):
    t = '<span class="tag">Answer</span>'
    if elapsed:
        t += f'<span class="tag t2">{elapsed:.1f}s</span>'
    return f'<div class="atags">{t}</div>'


def thinking_html(label):
    return f'<div class="thinking"><span></span><span></span><span></span><em>{label}</em></div>'


def render_message(m, idx):
    if m["role"] == "user":
        with st.container(key=f"u_{idx}"):
            st.markdown(m["content"])
    else:
        with st.container(key=f"a_{idx}"):
            st.markdown(tags_html(m.get("elapsed")), unsafe_allow_html=True)
            st.markdown(m["content"])
            if m.get("sources"):
                st.markdown(sources_html(m["sources"]), unsafe_allow_html=True)


question = st.chat_input("Ask a question about your documents")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

# The chat lives in a container created here, but it is filled in LAST. That way the library and
# "how it works" sections below are drawn first and do not fade out while an answer is generated.
chat_area = st.container()

# ---------- library ----------
def cover(i, f):
    st_ = stats.get(f.name)
    label = f"Page 1 preview" if f.name in excerpts else "Indexed"
    ex = excerpt(f.name, 700)
    body = f'<div class="cex">{ex}</div>' if ex else ""
    return (
        f'<div class="cov{"" if ex else " noex"}" style="background:{COVERS[i % len(COVERS)]}">'
        f'<span class="cchip">{label}</span>{body}</div>'
    )


docs_html = "".join(
    f'<div class="doc">{cover(i, f)}<div class="sline"><div class="sn">{i + 1:02d}.</div>'
    f'<div class="sm"><b>{esc(pretty(f.stem))}</b><span>'
    + (
        f'{stats[f.name]["pages"]} pages · {stats[f.name]["chunks"]} passages'
        if f.name in stats
        else "indexed"
    )
    + "</span></div></div></div>"
    for i, f in enumerate(doc_files)
)
st.markdown(
    f"""
<div id="library"></div>
<div class="sechead"><div class="stitle">Your <em>library</em></div>
  <div class="tl"><span>01.</span><i><b></b></i><span>{max(n_docs, 1):02d}.</span></div></div>
<div class="lib">{docs_html}</div>
<div class="bigfoot"><div class="bl"><b>Chunked 1000 / 150</b><span>Embeddings all-MiniLM-L6-v2</span>
<span>Vector store FAISS</span><span>LLM {provider}</span></div><div class="bw">Library</div></div>

<div id="how"></div>
<div class="sechead"><div class="stitle">How it <em>works</em></div><div class="tl"><i><b></b></i></div></div>
<div class="how">
  <div class="hc l"><div class="hn">01</div><div class="ht2">Ingest</div><div class="hd">Documents are split into overlapping passages,
    turned into vectors with a sentence-embedding model, and stored in a FAISS index.</div></div>
  <div class="hc"><div class="hn">02</div><div class="ht2">Retrieve</div><div class="hd">Your question becomes a vector too, and the
    most similar passages are pulled from the index in milliseconds.</div></div>
  <div class="hc d"><div class="hn">03</div><div class="ht2">Answer</div><div class="hd">The language model answers using only those
    passages and cites them. If the answer isn't there, it says so instead of guessing.</div></div>
</div>
<div class="footnote">Built with LangChain, FAISS and Streamlit.</div>
""",
    unsafe_allow_html=True,
)

# ---------- chat (filled last) ----------
with chat_area:
    if not st.session_state.messages and not question:
        st.markdown('<div class="try">Try <em>asking</em></div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if (e1 if i % 2 == 0 else e2).button(q, use_container_width=True, key=f"ex{i}"):
                st.session_state.pending = q
                st.rerun()

    for i, m in enumerate(st.session_state.messages):
        render_message(m, i)

    if question:
        user_msg = {"role": "user", "content": question}
        st.session_state.messages.append(user_msg)
        render_message(user_msg, len(st.session_state.messages) - 1)

        start = time.time()
        text, sources = "", []
        with st.container(key=f"a_{len(st.session_state.messages)}"):
            tags = st.empty()
            tags.markdown(tags_html(), unsafe_allow_html=True)
            body = st.empty()
            body.markdown(thinking_html("Searching documents"), unsafe_allow_html=True)
            try:
                docs, stream = answer_stream(question, store, llm, k=top_k)
                sources = [
                    {"name": d.metadata.get("source_name"), "page": d.metadata.get("page_label"), "text": d.page_content}
                    for d in docs
                ]
                st.markdown(sources_html(sources), unsafe_allow_html=True)  # shown right away
                body.markdown(thinking_html("Writing the answer"), unsafe_allow_html=True)
                for piece in stream:  # the answer appears as it is written
                    text += piece
                    body.markdown(text)
            except Exception as e:  # missing API key, rate limit, Ollama off...
                text = f"Error: {e}"
                body.markdown(text)
            elapsed = time.time() - start
            tags.markdown(tags_html(elapsed), unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": text, "sources": sources, "elapsed": elapsed}
        )