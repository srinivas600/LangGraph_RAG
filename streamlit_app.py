import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict
import uuid

class StreamlitChatbot:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = None
        if "sessions" not in st.session_state:
            st.session_state.sessions = []
        if "selected_session" not in st.session_state:
            st.session_state.selected_session = None

    def load_sessions(self):
        """Load all available sessions"""
        try:
            response = requests.get(f"{self.api_base_url}/sessions")
            if response.status_code == 200:
                return response.json()
            else:
                st.error("Failed to load sessions")
                return []
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API server. Please make sure it's running.")
            return []
        except Exception as e:
            st.error(f"Error loading sessions: {e}")
            return []

    def create_new_session(self):
        """Create a new chat session"""
        try:
            response = requests.post(f"{self.api_base_url}/sessions")
            if response.status_code == 200:
                result = response.json()
                st.session_state.current_session_id = result["session_id"]
                st.session_state.messages = []
                st.success("New session created!")
                st.rerun()
            else:
                st.error("Failed to create new session")
        except Exception as e:
            st.error(f"Error creating session: {e}")

    def load_chat_history(self, session_id: str):
        """Load chat history for a session"""
        try:
            response = requests.get(f"{self.api_base_url}/sessions/{session_id}/history")
            if response.status_code == 200:
                history = response.json()
                messages = []
                for msg in history:
                    messages.append({
                        "role": msg["message_type"],
                        "content": msg["message"],
                        "timestamp": msg["timestamp"]
                    })
                return messages
            else:
                st.error("Failed to load chat history")
                return []
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return []

    def send_message(self, message: str, session_id: str = None):
        """Send a message to the chatbot"""
        try:
            payload = {
                "message": message,
                "session_id": session_id
            }
            response = requests.post(f"{self.api_base_url}/chat", json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error sending message: {e}")
            return None

    def submit_feedback(self, transaction_id: str, feedback: str, feedback_text: str = None):
        """Submit feedback for a response"""
        try:
            payload = {
                "transaction_id": transaction_id,
                "feedback": feedback,
                "feedback_text": feedback_text
            }
            response = requests.post(f"{self.api_base_url}/feedback", json=payload)
            if response.status_code == 200:
                st.toast("Feedback submitted successfully!")
            else:
                st.error("Failed to submit feedback")
        except Exception as e:
            st.error(f"Error submitting feedback: {e}")

    def delete_session(self, session_id: str):
        """Delete a session"""
        try:
            response = requests.delete(f"{self.api_base_url}/sessions/{session_id}")
            if response.status_code == 200:
                st.success("Session deleted successfully!")
                if st.session_state.current_session_id == session_id:
                    st.session_state.current_session_id = None
                    st.session_state.messages = []
                st.rerun()
            else:
                st.error("Failed to delete session")
        except Exception as e:
            st.error(f"Error deleting session: {e}")

    def render_sidebar(self):
        """Render the sidebar for session management"""
        with st.sidebar:
            st.title("🤖 Agentic RAG Chatbot")
            st.markdown("---")
            st.subheader("Sessions")

            sessions = self.load_sessions()
            st.session_state.sessions = sessions

            if st.button("➕ New Session", use_container_width=True):
                self.create_new_session()

            if sessions:
                session_options = {f"{s['session_name']} ({s['message_count']} msgs)": s['session_id'] for s in sessions}
                
                # Safely determine the index of the current session
                session_ids = list(session_options.values())
                try:
                    current_index = session_ids.index(st.session_state.current_session_id)
                except (ValueError, IndexError):
                    current_index = 0

                selected_display = st.selectbox(
                    "Select Session:",
                    options=list(session_options.keys()),
                    index=current_index
                )

                selected_session_id = session_options[selected_display]

                if selected_session_id != st.session_state.current_session_id:
                    st.session_state.current_session_id = selected_session_id
                    st.session_state.messages = self.load_chat_history(selected_session_id)
                    st.rerun()

                if st.button("🗑️ Delete Selected Session", use_container_width=True):
                    if st.session_state.current_session_id:
                        self.delete_session(st.session_state.current_session_id)

            st.markdown("---")
            self.render_document_management()

    def render_document_management(self):
        """Render the document management section in the sidebar"""
        st.subheader("Knowledge Base")
        try:
            doc_response = requests.get(f"{self.api_base_url}/documents/count")
            if doc_response.status_code == 200:
                doc_count = doc_response.json()["document_count"]
                st.metric("Documents", doc_count)
        except:
            st.metric("Documents", "Error")

        with st.expander("Add Document"):
            new_doc = st.text_area("Document Content", placeholder="Enter document text...")
            if st.button("Add Document"):
                if new_doc.strip():
                    try:
                        payload = {"content": new_doc.strip()}
                        response = requests.post(f"{self.api_base_url}/documents", json=payload)
                        if response.status_code == 200:
                            st.success("Document added successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to add document")
                    except Exception as e:
                        st.error(f"Error adding document: {e}")

    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("💬 Chat Interface")

        if st.session_state.messages:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant" and "transaction_id" in message:
                        self.render_feedback_buttons(i, message["transaction_id"])

        if prompt := st.chat_input("What would you like to know?"):
            self.handle_chat_submission(prompt)

    def render_feedback_buttons(self, index: int, transaction_id: str):
        """Render feedback buttons for an assistant message"""
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"pos_{index}"):
                self.submit_feedback(transaction_id, "positive")
        with col2:
            if st.button("👎", key=f"neg_{index}"):
                self.submit_feedback(transaction_id, "negative")

    def handle_chat_submission(self, prompt: str):
        """Handle user chat input submission"""
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.current_session_id is None:
                    self.create_new_session()
                
                response_data = self.send_message(prompt, st.session_state.current_session_id)
                if response_data:
                    self.handle_successful_response(response_data)
                else:
                    st.error("Failed to get response from the chatbot")

    def handle_successful_response(self, response_data: Dict):
        """Handle a successful response from the chatbot API"""
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = response_data["session_id"]

        st.write(response_data["response"])
        assistant_message = {
            "role": "assistant",
            "content": response_data["response"],
            "transaction_id": response_data["transaction_id"]
        }
        st.session_state.messages.append(assistant_message)

        with st.expander("ℹ️ Response Details"):
            st.write(f"**Route taken:** {response_data['route_taken']}")
            st.write(f"**Transaction ID:** {response_data['transaction_id']}")
            if response_data.get("retrieved_documents"):
                st.write("**Retrieved Documents:**")
                for i, doc in enumerate(response_data["retrieved_documents"], 1):
                    st.write(f"{i}. {doc['content'][:200]}... (Score: {doc['score']:.3f})")

        st.rerun()

    def render_footer(self):
        """Render the footer"""
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "Powered by LangGraph, OpenAI GPT-4o, Chroma DB, and DeepEval"
            "</div>",
            unsafe_allow_html=True
        )
        if st.session_state.current_session_id:
            st.sidebar.info(f"Current Session: {st.session_state.current_session_id[:8]}...")

    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Agentic RAG Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.render_sidebar()
        self.render_chat_interface()
        self.render_footer()

if __name__ == "__main__":
    app = StreamlitChatbot()
    app.run()