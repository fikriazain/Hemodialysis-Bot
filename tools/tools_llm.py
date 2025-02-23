from langchain.tools import BaseTool, StructuredTool, tool
from langchain.utilities import GoogleSerperAPIWrapper
import requests
import random
import json
# from tools import db, reranker_model, model_llm_rag
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from deep_translator import GoogleTranslator
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = PROMPT_TEMPLATE = """### Instruction:
Your job is to answer the question based on the given pieces of information. All you have to do is answer the question. Not all of the information provided may be relevant to the question. the answer you create must be logical. Each piece of information will be separated by '---'.

### Example:
Question: What are the benefits of regular exercise for cardiovascular health?
---

Research published in the Journal of the American Heart Association indicates that regular exercise can reduce the risk of coronary heart disease by up to 30%. Physical activity helps strengthen heart muscles, improve blood circulation, and lower blood pressure.

---

Although exercise has many benefits, it is important to do it correctly to avoid injuries. Warming up before exercising and cooling down afterwards are highly recommended. Additionally, the type of exercise chosen should match the individual's physical condition to avoid unwanted risks.

---

According to a study from the Mayo Clinic, people who exercise regularly have better cholesterol levels and tend to have a healthier weight. Exercise can also increase insulin sensitivity and help regulate blood sugar levels, which are important factors in maintaining heart health.

---

Answer:
Regular physical exercise has several benefits for cardiovascular health. Firstly, it can reduce the risk of coronary heart disease by up to 30%, as it strengthens the heart muscles, improves blood circulation, and lowers blood pressure. Secondly, individuals who exercise regularly tend to have better cholesterol levels and a healthier weight, which are crucial for heart health. Additionally, regular exercise can increase insulin sensitivity and help regulate blood sugar levels, further contributing to cardiovascular well-being.

### Another example:
Question: What are the benefits of a fiber-rich diet for digestive health?
---

A fiber-rich diet is known to prevent constipation by increasing stool bulk and softness, making it easier to pass. Fiber also helps maintain gut health by promoting the growth of beneficial bacteria in the digestive system.

---

High-fiber foods such as fruits, vegetables, and whole grains are not only good for digestion but can also help control blood sugar levels and lower cholesterol. Soluble fiber in these foods helps slow down sugar absorption and binds cholesterol in the intestines.

---

Some studies suggest that a high-fiber diet can reduce the risk of colorectal cancer. Fiber helps speed up the elimination of carcinogenic substances from the colon, reducing the exposure time of colon cells to harmful materials.

---

Answer:
A diet rich in fiber has multiple benefits for digestive health. It can prevent constipation by increasing stool bulk and softness, making it easier to pass. Fiber also promotes gut health by encouraging the growth of beneficial bacteria in the digestive system. Additionally, high-fiber foods such as fruits, vegetables, and whole grains can help control blood sugar levels and lower cholesterol. Soluble fiber in these foods slows sugar absorption and binds cholesterol in the intestines. Furthermore, a high-fiber diet can reduce the risk of colorectal cancer by speeding up the removal of carcinogenic substances from the colon, thereby reducing the exposure time of colon cells to harmful materials.

### Input
Question: {question}
---

{context}

---

Answer:
"""

@tool
def send_emergency_message_to_medic(query: str) -> str: 
    """This function is used to send a message containing user symptoms to the medic where the symptoms are related to emergency cases. 
    You must give the query semantically the same with the user input,
    You can ONLY run this function ONE time, then you must run the 'search_hemodialysis_information' tools to get user symptoms explanation."""
    url = "http://127.0.0.1:8000/message/get_message/"

    user_id = str(random.randint(1, 100))

    data = {
        "message": query,
        "user_id": user_id
    }

    #Turn data into json for the request
    data = json.dumps(data)

    response = requests.post(url, data=data)
    return ' Success sending message. Please provide search query for the symtomps that patient has.\n'

@tool
def search_information_for_question(query: str) -> str:
    """Function that searches for information based on the user query. You must use this function if there are questions related to medical topics. 
    # The query is the message that the patient send to Panda, YOU MUST NOT CHANGE IT."""
    # compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=db
    # )

    # query_translate = GoogleTranslator(source='english', target='id').translate(query)
    # results = compression_retriever.invoke(query)
    # target = "\n\n---\n\n".join([doc.page_content for doc in results])
    # context_text = GoogleTranslator(source='id', target='english').translate(target)
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_translate)
    # print(target)
    # result = model_llm_rag.invoke(prompt)
    # return GoogleTranslator(source='id', target='english').translate(result)
    return GoogleSerperAPIWrapper().run(query)

# @tool
# def search_medic_info(query: str) -> str:
#     results = db.similarity_search_with_relevance_scores(query, k=3)
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     return context_text

# @tool
# def search_medic_info(query: str) -> str: 
#     """Function that searches for medical information based on the user query. The query is the message that the patient send to Panda, YOU MUST NOT CHANGE IT."""
#     return GoogleSerperAPIWrapper().run(query)