import json
import sys
from langchain.schema import Document
from langchain import callbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import pdfplumber
import requests
from dotenv import load_dotenv
from uuid import uuid4
import io,os,re
from concurrent.futures import ThreadPoolExecutor

#==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
#==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Call_Center_Operation_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Consulting_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Content_Production_Service_Contract_(Request_Form).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Customer_Referral_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Draft_Editing_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Graphic_Design_Production_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/software_development_outsourcing_contracts.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Technical_Verification_(PoC)_Contract.pdf",
]
#==============================================================================


#==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
#==============================================================================
def rag_implementation(question: str) -> str:
    llm = ChatOpenAI(model=model)
    pdf_list =[]
    category = []
    def process_pdf(file_url):
        response = requests.get(file_url)
        response.raise_for_status()
        pdf_content = ""
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            # PDFの各ページに対して処理を行う
            for page in pdf.pages:
                # ページからテキストデータを抽出
                text = page.extract_text()
                # 抽出したテキストデータをpdf_contentに格納
                pdf_content += text

        # 改行コードで区切ってリストにする
        pdf_content_list = pdf_content.split("\n")
        pdf_content_list = [item.replace(" ", "").replace("\u3000", "") for item in pdf_content_list]

        pdf_info_json = {}
        filename = os.path.basename(file_url)
        pdf_info_json["file_name"] = filename[:filename.rfind('.')]
        pdf_info_json["length"] = len(pdf_content)
        pdf_info_json["title"] = pdf_content_list[0]
        category.append(pdf_content_list[0])
        pdf_info_json["content0"] = ""

        i = 0
        for pdf_content_list_item in pdf_content_list:
            if re.search(r"第\d+条", pdf_content_list_item) and "（" in pdf_content_list_item and len(pdf_content_list_item) < 30 and pdf_content_list_item.endswith("）"):
                i += 1
                pdf_info_json["content" + str(i)] = pdf_content_list_item
            else:
                pdf_info_json["content" + str(i)] += pdf_content_list_item

        return pdf_info_json

    # スレッドプールを使用してPDFの処理を実行
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_file_urls)
        pdf_list = list(results)
    pass
    #【PDFリスト】→【langchainドキュメント化】
    document_list = []
    #ファイルごとに処理
    for pdf_file in pdf_list:
        #章ごとに処理
        for key in list(pdf_file.keys()):

            if "content" in key:
                faiss_document = Document(page_content=pdf_file["title"] + pdf_file[key], metadata={
                    "file_name": pdf_file["file_name"],
                    "title": pdf_file["title"]
                })
                document_list.append(faiss_document)
    #【langchainドキュメント】→【ベクタストア化】
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        # persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
    )
    uuids = [str(uuid4()) for _ in range(len(document_list))]
    vector_store.add_documents(documents=document_list, ids=uuids)


    category_template = ChatPromptTemplate(
        [
            ("system",
                '''与えられた質問文がどのカテゴリーに含まれるか答えてください。回答はカテゴリーのみ答えてください。
                以下カテゴリーに分類できない場合は、
                その他
                と回答してください。
                カテゴリー
                {category}'''
            ),
            ("user", "{user_input}")
        ]
    )

    category_message = category_template.format_messages(category=category, user_input=question)
    category_output = llm.invoke(category_message)

    
    if category_output.content == "その他":
        print(category_output.content)
        docs_and_scores = vector_store.similarity_search_with_score(question)
    else:
        docs_and_scores = vector_store.similarity_search_with_score(question, filter={"title": category_output.content})
    
    similar_point = 0.95  # この数値を以下を引用対象とする
    filtered_docs_and_scores = [item for item in docs_and_scores if item[1] <= similar_point]  # ステップ2: フィルタリング
    inputdata = ""
    for i,doc_and_score in enumerate(filtered_docs_and_scores):
        inputdata +=f'''###{i+1}件目引用資料###\n
            {doc_and_score[0].page_content}\n
            ############\n
        '''

    print(filtered_docs_and_scores)
    
    # プロンプトテンプレート
    chat_template = ChatPromptTemplate(
        [
            ("system",
                '''あなたは法務の質問に回答するチャットボットです。質問と関連する内容から回答してください。該当する資料から明確な回答を見つけられない場合、わかりませんと回答します。
                以下、引用資料
                {inputdata}''',
            ),
            ("user", "{user_input}"),
        ]
    )

    
    

    messages = chat_template.format_messages(inputdata=inputdata,user_input=question)
    output =llm.invoke(messages)
    print("以下テスト回答##################")
    print(output)
    print("以下テスト回答##################")
    answer = output.content
    return answer
#==============================================================================


#==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
#==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        run_id = cb.traced_runs[0].id

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
#==============================================================================
