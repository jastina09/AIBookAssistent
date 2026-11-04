import html
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import main

st.set_page_config(page_title="AI Book Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
.hero {
    padding: 28px 32px;
    border-radius: 18px;
    background: linear-gradient(120deg, #6a5acd 0%, #3b82f6 55%, #06b6d4 100%);
    color: white;
    margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.2rem; color: white; padding: 0; }
.hero p { margin: 8px 0 0 0; font-size: 1.05rem; opacity: 0.95; }
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    background: rgba(106, 90, 205, 0.18);
    font-size: 0.8rem;
    margin-right: 6px;
}
.book-title { font-size: 1.1rem; font-weight: 700; margin: 8px 0 2px 0; }
.book-author { opacity: 0.7; font-size: 0.9rem; margin-bottom: 8px; }
.stTabs [data-baseweb="tab"] { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

genre_emoji = {
    "антиутопия": "🏙️",
    "фантастика": "🚀",
    "фэнтези": "🐉",
    "детектив": "🕵️",
    "классика": "🏛️",
    "повесть": "🌄",
    "притча": "🌟",
    "психология": "🧠",
    "саморазвитие": "🌱",
    "нон-фикшн": "🌍",
}
books_by_title = {b["title"]: b for b in main.books}
genres = sorted(set(b["genre"] for b in main.books))


@st.cache_resource
def load_store():
    return main.create_vector_store()


store = load_store()
recommend_chain = main.make_recommend_chain(store)
explain_chain = main.make_explain_chain(store)
card_chain = main.make_card_chain(store)

if "history" not in st.session_state:
    st.session_state.history = []


def set_value(key, value):
    st.session_state[key] = value


def example_buttons(examples, key):
    st.caption("Попробуй пример:")
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        cols[i].button(ex, key=f"{key}_example_{i}", on_click=set_value, args=(key, ex))


def show_error(e):
    st.error("Не получилось получить ответ. Проверь, что Ollama запущена и модели скачаны.")
    st.caption(str(e))


with st.sidebar:
    st.header("📚 О проекте")
    st.write("Бот ищет подходящие книги в своей базе знаний и объясняет, почему они тебе подойдут.")

    c1, c2 = st.columns(2)
    c1.metric("Книг", len(main.books))
    c2.metric("Жанров", len(genres))

    st.subheader("Как это работает")
    st.markdown(
        "1. Ты пишешь запрос\n"
        "2. Бот ищет похожие книги в базе (поиск по смыслу)\n"
        "3. Языковая модель пишет ответ только по найденным книгам"
    )
    st.divider()
    if st.button("🗑️ Очистить чат"):
        st.session_state.history = []

st.markdown("""
<div class="hero">
    <h1>📚 AI Book Assistant</h1>
    <p>Помогу выбрать книгу по твоим интересам, объясню о чём она и подскажу, что читать дальше.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "✨ Подбор книг", "📖 О книге", "💬 Чат", "🗂️ Каталог"
])

with tab1:
    st.subheader("Расскажи, что тебе нравится")
    example_buttons(
        ["люблю фантастику", "хочу детектив с загадкой", "книги про психологию"],
        "interests",
    )
    interests = st.text_area(
        "Твои интересы",
        key="interests",
        height=100,
        placeholder="Например: люблю книги про будущее и технологии",
    )
    if st.button("✨ Подобрать книги", type="primary"):
        if interests.strip() == "":
            st.warning("Сначала напиши, что тебе интересно")
        else:
            try:
                with st.container(border=True):
                    with st.spinner("Ищу подходящие книги..."):
                        st.write_stream(recommend_chain.stream(interests))
            except Exception as e:
                show_error(e)

with tab2:
    st.subheader("Узнай больше о книге")
    title = st.selectbox(
        "Выбери книгу",
        list(books_by_title.keys()),
        format_func=lambda t: f"{t} — {books_by_title[t]['author']}",
    )
    if st.button("📖 Рассказать о книге", type="primary"):
        base = books_by_title[title]
        try:
            with st.spinner("Готовлю карточку книги..."):
                card = card_chain.invoke(title)

            emoji = genre_emoji.get(base["genre"], "📘")
            with st.container(border=True):
                st.markdown(f"### {emoji} {card.title}")
                st.caption(f"{card.author} • {card.genre} • {base['year']}")
                st.write(card.summary)
                st.info(f"Кому понравится: {card.for_whom}")

            st.markdown("#### Подробнее")
            with st.container(border=True):
                st.write_stream(explain_chain.stream(title))
        except Exception as e:
            show_error(e)

with tab3:
    st.subheader("Чат с помощником")

    if not st.session_state.history:
        st.info("Задай любой вопрос про книги, например: «Какую книгу почитать на выходные?»")

    for m in st.session_state.history:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(m.content)

    question = st.chat_input("Спроси что-нибудь про книги...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Думаю..."):
                    response = main.chat_with_history(store, st.session_state.history, question)
                st.write(response.content)

            st.session_state.history.append(HumanMessage(content=question))
            st.session_state.history.append(AIMessage(content=response.content))

            with st.expander("Информация об ответе"):
                st.write("Модель:", response.response_metadata.get("model"))
                st.write("Токены:", response.usage_metadata)
        except Exception as e:
            show_error(e)

with tab4:
    st.subheader("Книги в базе знаний")
    col_a, col_b = st.columns([2, 1])
    search = col_a.text_input("Поиск по названию или автору")
    chosen = col_b.multiselect("Жанр", genres)

    shown = []
    for b in main.books:
        text = (b["title"] + " " + b["author"]).lower()
        if search.lower() in text and (not chosen or b["genre"] in chosen):
            shown.append(b)

    if not shown:
        st.info("Ничего не найдено")

    cols = st.columns(3)
    for i, b in enumerate(shown):
        with cols[i % 3]:
            with st.container(border=True):
                emoji = genre_emoji.get(b["genre"], "📘")
                st.markdown(
                    f'<span class="tag">{emoji} {b["genre"]}</span><span class="tag">{b["year"]}</span>'
                    f'<div class="book-title">{html.escape(b["title"])}</div>'
                    f'<div class="book-author">{html.escape(b["author"])}</div>',
                    unsafe_allow_html=True,
                )
                st.write(b["description"])

st.divider()
st.caption("Проект на LangChain: RAG (Document + поиск по смыслу) и Streamlit")