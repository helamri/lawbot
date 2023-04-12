from flask import Flask, request, render_template
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

app = Flask(__name__)

embeddings = OpenAIEmbeddings(openai_api_key="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")
pinecone.init(
    api_key="04a90dac-d0f2-4690-87dc-2c4b8ed216c5",
    environment="us-east1-gcp"
)
index_name = "laws"
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key="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", max_tokens=1900)

template = "You are a helpful Moroccan laws expert, Answer the questions in detail in the language the question was asked. {documents}"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=llm, prompt=chat_prompt)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        docs = docsearch.similarity_search(query, include_metadata=True)
        result = chain.run(documents=docs, question=query)
        return render_template('index.html', result=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
