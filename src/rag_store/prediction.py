import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.upstash import UpstashVectorStore

# Handle imports for both src/ and root directory usage
try:
    from src.prompts.rag_prompt import RAG_PROMPT_TEMPLATE
    from src.callbacks.streamlit_callback import StreamHandler
except ImportError:
    # If running from src/ directory, try parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from prompts.rag_prompt import RAG_PROMPT_TEMPLATE
        from callbacks.streamlit_callback import StreamHandler
    except ImportError:
        # Fallback: define inline
        RAG_PROMPT_TEMPLATE = """
        Your task is to answer questions by using a given context.

        Don't invent anything that is outside of the context.
        Answer in at least 350 characters.

        %CONTEXT%
        {context}

        %Question%
        {question}

        Hint: Do not copy the context. Use your own words

        Answer:
        """

        
        # Simple fallback StreamHandler
        from langchain.callbacks.base import BaseCallbackHandler
        
        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container):
                self.container = container
                self.text = ""
                
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text += token
                if hasattr(self.container, 'markdown'):
                    self.container.markdown(self.text)
                else:
                    print(token, end="", flush=True)


load_dotenv()

class RAG:
    def __init__(self, chat_box, embeddings):
        self.chat_box = chat_box
        self.embeddings = embeddings
        self.set_llm()
        
        self.vectorstore = UpstashVectorStore(embedding=embeddings)

    def set_llm(self):
        """Initialize the language model with appropriate streaming settings"""
        if self.chat_box:
            # Streamlit mode with streaming
            try:
                chat_container = self.chat_box.container().empty()
                stream_handler = StreamHandler(chat_container)
                callbacks = [stream_handler]
                streaming = True
            except:
                # Fallback if Streamlit container setup fails
                callbacks = []
                streaming = False
        else:
            # Terminal mode without streaming
            callbacks = []
            streaming = False
            
        self.llm = ChatOpenAI(
            max_tokens=400,
            streaming=streaming,
            callbacks=callbacks,
            model="gpt-3.5-turbo",
            temperature=0.1,  # Lower temperature for more consistent answers
        )

    def get_context(self, query, k=4):
        """Get relevant context from vector store"""
        try:
            # Use the official similarity_search_with_score method
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            context = ""

            for doc, score in results:
                # Add score and content with separator
                context += f"[Relevance: {score:.3f}]\n{doc.page_content}\n{'='*50}\n"
            
            return context, results
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return "", []

    @staticmethod
    def get_prompt(question, context):
        """Format the RAG prompt with question and context"""
        return RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    def predict(self, query):
        """Main prediction method for RAG"""
        try:
            # Get relevant context from vector store
            context, source_documents = self.get_context(query)
            
            if not context.strip():
                context = "No relevant documents found in the knowledge base."
            
            # Format prompt with context
            prompt = self.get_prompt(query, context)
            
            # Generate answer using LLM
            answer = self.llm.invoke(prompt)
            
            prediction = {
                "answer": answer,
                "source_documents": source_documents,
                "context_used": context,
            }
            return prediction
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "source_documents": [],
                "context_used": "",
            }

    def test_connection(self):
        """Test if the vector store connection is working"""
        try:
            results = self.vectorstore.similarity_search("test", k=1)
            print(f"✅ Vector store connection working. Found {len(results)} documents.")
            return len(results) > 0
        except Exception as e:
            print(f"❌ Vector store connection failed: {e}")
            return False
