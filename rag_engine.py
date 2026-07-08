from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


class CouncilRAG:
    def __init__(self):
        # ✅ Improved embedding model (better accuracy)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2"
        )

        self.vector_db = None
        self.vector_retriever = None
        self.keyword_retriever = None

    def build_index(self, data_path="data/"):
        # =============================
        # LOAD DOCUMENTS
        # =============================
        loader = DirectoryLoader(
            data_path,
            glob="*.txt",
            loader_cls=TextLoader
        )

        documents = loader.load()

        # =============================
        # IMPROVED TEXT SPLITTING
        # =============================
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,   # ✅ FIXED (was 500)
            separators=["\n\n", "\n", " ", ""]
        )

        docs = text_splitter.split_documents(documents)

        # =============================
        # VECTOR DATABASE
        # =============================
        self.vector_db = Chroma.from_documents(
            docs,
            self.embeddings
        )

        self.vector_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 4}   # slightly more for better recall
        )

        # =============================
        # BM25 RETRIEVER
        # =============================
        self.keyword_retriever = BM25Retriever.from_documents(docs)
        self.keyword_retriever.k = 4

    def query(self, user_input):
        if not self.vector_retriever or not self.keyword_retriever:
            return []

        # =============================
        # HYBRID RETRIEVAL
        # =============================
        vector_docs = self.vector_retriever.invoke(user_input)
        keyword_docs = self.keyword_retriever.invoke(user_input)

        combined = vector_docs + keyword_docs

        # =============================
        # REMOVE DUPLICATES
        # =============================
        seen = set()
        unique_docs = []

        for doc in combined:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        # =============================
        # SIMPLE RANKING STRATEGY
        # =============================
        # Rank by chunk length (proxy for information richness)
        ranked_docs = sorted(
            unique_docs,
            key=lambda x: len(x.page_content),
            reverse=True   # longer = more useful
        )

        # =============================
        # RETURN TOP RESULTS (CONTROL CONTEXT SIZE)
        # =============================
        return ranked_docs[:3]   # ✅ reduced for better LLM performance