text = '\n\nParticipants:\n\nJerry: Loves movies and is a bit of a klutz.\nSamantha: Enthusiastic about food and always trying new restaurants.\nBarry: A nature lover, but always manages to get lost.\nJerry: Hey, guys! You won\'t believe what happened to me at the Times Square AMC theater. I tripped over my own feet and spilled popcorn everywhere! 🍿💥\n\nSamantha: LOL, that\'s so you, Jerry! Was the floor buttery enough for you to ice skate on after that? 😂\n\nBarry: Sounds like a regular Tuesday for you, Jerry. Meanwhile, I tried to find that new hiking trail in Central Park. You know, the one that\'s supposed to be impossible to get lost on? Well, guess what...\n\nJerry: You found a hidden treasure?\n\nBarry: No, I got lost. AGAIN. 🧭🙄\n\nSamantha: Barry, you\'d get lost in your own backyard! But speaking of treasures, I found this new sushi place in Little Tokyo. "Samantha\'s Sushi Symphony" it\'s called. Coincidence? I think not!\n\nJerry: Maybe they named it after your ability to eat your body weight in sushi. 🍣\n\nBarry: How do you even FIND all these places, Samantha?\n\nSamantha: Simple, I don\'t rely on Barry\'s navigation skills. 😉 But seriously, the wasabi there was hotter than Jerry\'s love for Marvel movies!\n\nJerry: Hey, nothing wrong with a little superhero action. By the way, did you guys see the new "Captain Crunch: Breakfast Avenger" trailer?\n\nSamantha: Captain Crunch? Are you sure you didn\'t get that from one of your Saturday morning cereal binges?\n\nBarry: Yeah, and did he defeat his arch-enemy, General Mills? 😆\n\nJerry: Ha-ha, very funny. Anyway, that sushi place sounds awesome, Samantha. Next time, let\'s go together, and maybe Barry can guide us... if we want a city-wide tour first.\n\nBarry: As long as we\'re not hiking, I\'ll get us there... eventually. 😅\n\nSamantha: It\'s a date! But Jerry, you\'re banned from carrying any food items.\n\nJerry: Deal! Just promise me no wasabi challenges. I don\'t want to end up like the time I tried Sriracha ice cream.\n\nBarry: Wait, what happened with Sriracha ice cream?\n\nJerry: Let\'s just say it was a hot situation. Literally. 🔥\n\nSamantha: 🤣 I still have the video!\n\nJerry: Samantha, if you value our friendship, that video will never see the light of day.\n\nSamantha: No promises, Jerry. No promises. 🤐😈\n\nBarry: I foresee a fun weekend ahead! 🎉\n\n'
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

def test():
    if False:
        return 10
    embeddings = OpenAIEmbeddings()
    dataset_path = 'hub://testingacc2/data'
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = text_splitter.split_text(dataset_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(pages)
    embeddings = OpenAIEmbeddings()
    db = DeepLake.from_documents(texts, embeddings, dataset_path=dataset_path, overwrite=True, runtime={'tensor_db': True})
    assert db.vectorstore.exec_option == 'tensor_db'
    db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 4
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=retriever, return_source_documents=False)
    query = 'what did Jerry do?'
    ans = qa({'query': query})