from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field


CHAT_MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"

llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0.5,
    top_p=0.9,
    top_k=40,
    seed=42,
    num_predict=600,
    num_ctx=4096,
)


books = [
    {"title": "1984", "author": "Джордж Оруэлл", "genre": "антиутопия", "year": 1949,
     "description": "Мир, где государство следит за каждым шагом человека, а правда переписывается по приказу партии. Книга о свободе мысли и власти."},
    {"title": "451 градус по Фаренгейту", "author": "Рэй Брэдбери", "genre": "антиутопия", "year": 1953,
     "description": "Пожарный, который сжигает книги, вдруг начинает сомневаться в своей работе. О том, почему чтение делает людей свободнее."},
    {"title": "Дюна", "author": "Фрэнк Герберт", "genre": "фантастика", "year": 1965,
     "description": "Далёкая пустынная планета, борьба за редкую пряность и путь молодого героя. Политика, экология и религия в одной истории."},
    {"title": "Пикник на обочине", "author": "Аркадий и Борис Стругацкие", "genre": "фантастика", "year": 1972,
     "description": "После визита пришельцев остались опасные Зоны с загадочными артефактами. Сталкеры рискуют жизнью ради добычи оттуда."},
    {"title": "Хоббит", "author": "Дж. Р. Р. Толкин", "genre": "фэнтези", "year": 1937,
     "description": "Бильбо Бэггинс отправляется с гномами за сокровищами дракона. Лёгкое и тёплое приключение, хорошо подходит для начала знакомства с фэнтези."},
    {"title": "Убийство в Восточном экспрессе", "author": "Агата Кристи", "genre": "детектив", "year": 1934,
     "description": "В снежном заносе посреди пути в поезде убит пассажир. Эркюль Пуаро должен найти убийцу среди запертых в вагоне людей."},
    {"title": "Этюд в багровых тонах", "author": "Артур Конан Дойл", "genre": "детектив", "year": 1887,
     "description": "Первое знакомство Шерлока Холмса и доктора Ватсона. Загадочное убийство в пустом доме и знаменитый метод дедукции."},
    {"title": "Преступление и наказание", "author": "Фёдор Достоевский", "genre": "классика", "year": 1866,
     "description": "Бедный студент совершает убийство, считая, что имеет на это право, и мучается от чувства вины. Глубокая книга о совести и морали."},
    {"title": "Мастер и Маргарита", "author": "Михаил Булгаков", "genre": "классика", "year": 1967,
     "description": "Дьявол приезжает в Москву 30-х годов и переворачивает жизнь горожан. Рядом идёт история любви Мастера и Маргариты."},
    {"title": "Джамиля", "author": "Чингиз Айтматов", "genre": "повесть", "year": 1958,
     "description": "Короткая повесть о любви молодой женщины и раненого солдата Данияра на фоне киргизской деревни. Много красивых описаний природы."},
    {"title": "Маленький принц", "author": "Антуан де Сент-Экзюпери", "genre": "притча", "year": 1943,
     "description": "Лётчик в пустыне встречает мальчика с другой планеты. Простая сказка о дружбе, любви и том, что важно по-настоящему."},
    {"title": "Алхимик", "author": "Пауло Коэльо", "genre": "притча", "year": 1988,
     "description": "Пастух Сантьяго идёт в Египет за сокровищем, о котором ему приснился сон. История о том, как следовать своей мечте."},
    {"title": "Думай медленно... решай быстро", "author": "Даниэль Канеман", "genre": "психология", "year": 2011,
     "description": "Автор объясняет две системы мышления и показывает, в каких ошибках мы попадаемся при принятии решений."},
    {"title": "Атомные привычки", "author": "Джеймс Клир", "genre": "саморазвитие", "year": 2018,
     "description": "Практическая книга о том, как маленькие ежедневные действия формируют привычки и меняют жизнь."},
    {"title": "Человек в поисках смысла", "author": "Виктор Франкл", "genre": "психология", "year": 1946,
     "description": "Психолог рассказывает о своём опыте в концлагере и о том, как смысл жизни помогает выдержать любые трудности."},
    {"title": "Sapiens. Краткая история человечества", "author": "Юваль Ной Харари", "genre": "нон-фикшн", "year": 2011,
     "description": "История человека от древних времён до наших дней: как мы победили других видов, создали деньги, религии и государства."},
]


def make_documents():
    docs = []
    for b in books:
        text = f"{b['title']} ({b['author']}). Жанр: {b['genre']}. {b['description']}"
        doc = Document(
            page_content=text,
            metadata={"title": b["title"], "author": b["author"], "genre": b["genre"], "year": b["year"]},
        )
        docs.append(doc)
    return docs


def create_vector_store():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    store = InMemoryVectorStore(embeddings)
    store.add_documents(make_documents())
    return store


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)



recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты помощник по книгам. Отвечай по-русски. Рекомендуй книги только из контекста, ничего не выдумывай."),
    ("human", "Контекст:\n{context}\n\nИнтересы пользователя: {question}\n\n"
              "Посоветуй 3 книги из контекста и коротко объясни, почему они подойдут."),
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты помощник по книгам. Отвечай по-русски и используй только информацию из контекста."),
    ("human", "Контекст:\n{context}\n\nКнига: {question}\n\n"
              "Объясни простыми словами, о чём эта книга, какие в ней главные темы и кому её стоит читать."),
])

card_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты помощник по книгам. Заполни карточку книги по контексту, отвечай по-русски."),
    ("human", "Контекст:\n{context}\n\nНазвание книги: {question}"),
])

chat_system_text = (
    "Ты дружелюбный помощник по книгам. Отвечай по-русски и коротко. "
    "Если советуешь книги, бери их из контекста ниже."
)



class BookCard(BaseModel):
    title: str = Field(description="название книги")
    author: str = Field(description="автор книги")
    genre: str = Field(description="жанр")
    summary: str = Field(description="краткое содержание в двух предложениях")
    for_whom: str = Field(description="кому понравится эта книга")



def make_retrieval_part(store, k=4):
    retriever = store.as_retriever(search_kwargs={"k": k})
    return RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough(),
    )


def make_recommend_chain(store):
    return make_retrieval_part(store, k=4) | recommend_prompt | llm | StrOutputParser()


def make_explain_chain(store):
    return make_retrieval_part(store, k=1) | explain_prompt | llm | StrOutputParser()


def make_card_chain(store):
    structured_llm = llm.with_structured_output(BookCard)
    return make_retrieval_part(store, k=1) | card_prompt | structured_llm



def chat_with_history(store, history, question):
    # history - список HumanMessage и AIMessage
    found = store.similarity_search(question, k=3)
    context = format_docs(found)

    messages = [SystemMessage(content=chat_system_text + "\n\nКонтекст:\n" + context)]
    messages += history
    messages.append(HumanMessage(content=question))

    return llm.invoke(messages)



def run_tests():
    store = create_vector_store()
    chain = make_recommend_chain(store)

    print("invoke:")
    print(chain.invoke("люблю детективы"))

    print("\nbatch:")
    answers = chain.batch(["хочу фантастику", "что почитать по психологии"])
    for a in answers:
        print(a)
        print("-" * 40)

    print("\nstream:")
    for part in chain.stream("хочу что-нибудь из классики"):
        print(part, end="", flush=True)
    print()

    print("\nsimilarity search:")
    for doc in store.similarity_search("книги про дружбу", k=2):
        print(doc.metadata["title"])



def open_docs_when_ready():
    import socket
    import time
    import webbrowser

    while True:
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    webbrowser.open("http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    import threading
    import uvicorn

    threading.Thread(target=open_docs_when_ready, daemon=True).start()
    uvicorn.run("api:app", host="127.0.0.1", port=8000)