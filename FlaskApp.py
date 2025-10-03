from flask import Flask, request, jsonify
import json
import re
import os
import time
from io import BytesIO
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Third-party libraries for document processing
import PyPDF2
import docx

# Google Gemini SDK for LLM integration
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- Pydantic Models for Structured Output ---

class DocumentSummary(BaseModel):
    """Structured summary of a legal document"""
    document_type: str = Field(..., description="Type of legal document")
    parties_involved: List[str] = Field(default_factory=list, description="All parties mentioned in the document")
    key_dates: List[Dict[str, str]] = Field(default_factory=list, description="Important dates mentioned with context")
    primary_subjects: List[str] = Field(default_factory=list, description="Primary legal subjects covered")
    key_provisions: List[str] = Field(default_factory=list, description="Key legal provisions or clauses")
    critical_obligations: List[str] = Field(default_factory=list, description="Critical obligations mentioned")
    potential_issues: List[str] = Field(default_factory=list, description="Potential legal issues identified")
    summary: str = Field(..., description="Concise summary of the document")

class DocumentQuery(BaseModel):
    """Response to a specific query about the document"""
    direct_references: List[Dict[str, str]] = Field(default_factory=list, description="Direct references from document with page/section numbers")
    interpretation: str = Field(..., description="Legal interpretation of the query based on document")
    related_principles: List[str] = Field(default_factory=list, description="Related legal principles")
    strategic_insights: List[str] = Field(default_factory=list, description="Strategic insights for the lawyer")
    limitations: List[str] = Field(default_factory=list, description="Limitations or ambiguities in the document related to the query")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions based on the query")

# --- LegalDocumentAnalyzer Class (Migrated to Gemini) ---

class LegalDocumentAnalyzer:
    def __init__(self):
        # Initialize Gemini API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError(
                "API key not found. Please set the GEMINI_API_KEY environment variable."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize the Gemini model
        # Using gemini-1.5-pro for best performance with large documents
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        self.document_text = ""
        self.document_name = ""
        
    def extract_text_from_document(self, file_content: bytes, file_name: str) -> str:
        """Extract text from various document formats using in-memory file content"""
        try:
            file_extension = file_name.split('.')[-1].lower()
            self.document_name = file_name
            
            if file_extension == 'pdf':
                text = ""
                with BytesIO(file_content) as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page_text = reader.pages[page_num].extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                return text
                
            elif file_extension == 'docx':
                doc = docx.Document(BytesIO(file_content))
                text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
                
            elif file_extension == 'txt':
                with BytesIO(file_content) as file:
                    text = file.read().decode('utf-8')
                return text
                
            else:
                return f"Unsupported file format: .{file_extension}"
                
        except Exception as e:
            self.document_text = ""
            return f"Error extracting text: {str(e)}"
    
    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from response text that might contain markdown or extra text"""
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON object boundaries
        if '{' in text and '}' in text:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            return text[start_idx:end_idx]
        return text
    
    def summarize_document(self) -> DocumentSummary:
        """Generate a structured summary of the uploaded legal document using Gemini"""
        try:
            if not self.document_text:
                return DocumentSummary(
                    document_type="Unknown",
                    parties_involved=[],
                    key_dates=[],
                    primary_subjects=[],
                    key_provisions=[],
                    critical_obligations=[],
                    potential_issues=[],
                    summary="No document text available for analysis."
                )
            
            # Truncate document if too long (Gemini 1.5 Pro handles up to 1M tokens)
            max_text_length = 800000
            truncated_text = self.document_text[:max_text_length]
            if len(self.document_text) > max_text_length:
                truncated_text += "\n[Document truncated due to length...]"
            
            # Create the prompt for Gemini
            prompt = f"""You are an expert legal AI assistant specializing in analyzing legal documents.

Analyze the provided legal document and generate a comprehensive structured summary in JSON format.

DOCUMENT NAME: {self.document_name}

DOCUMENT CONTENT:
{truncated_text}

Provide your analysis as a JSON object with the following structure:
{{
  "document_type": "string - Type of legal document",
  "parties_involved": ["list of all parties mentioned"],
  "key_dates": [{{"date": "date string", "context": "what the date relates to"}}],
  "primary_subjects": ["list of primary legal subjects covered"],
  "key_provisions": ["list of key legal provisions or clauses"],
  "critical_obligations": ["list of critical obligations"],
  "potential_issues": ["list of potential legal issues identified"],
  "summary": "string - Concise overall summary of the document"
}}

Respond with ONLY the JSON object, no additional text or markdown formatting."""

            # Generate content using Gemini
            start_time = time.time()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=4096,
                )
            )
            
            response_time = time.time() - start_time
            
            # Extract and parse the response
            raw_response = response.text.strip()
            json_str = self._extract_json_from_response(raw_response)
            analysis_dict = json.loads(json_str)
            
            # Create and return Pydantic model instance
            return DocumentSummary(**analysis_dict)
            
        except (json.JSONDecodeError, ValidationError) as e:
            error_message = f"Error parsing response: {type(e).__name__}: {str(e)}"
            return DocumentSummary(
                document_type="Error",
                parties_involved=[],
                key_dates=[],
                primary_subjects=[],
                key_provisions=[],
                critical_obligations=[],
                potential_issues=[],
                summary=error_message
            )
        except Exception as e:
            return DocumentSummary(
                document_type="Error",
                parties_involved=[],
                key_dates=[],
                primary_subjects=[],
                key_provisions=[],
                critical_obligations=[],
                potential_issues=[],
                summary=f"Unknown error: {str(e)}"
            )
    
    def query_document(self, query: str) -> DocumentQuery:
        """Generate a response to a specific query about the document"""
        try:
            if not self.document_text:
                return DocumentQuery(
                    direct_references=[],
                    interpretation="No document loaded to analyze.",
                    related_principles=[],
                    strategic_insights=[],
                    limitations=["No document has been uploaded"],
                    recommended_actions=["Please upload a document first"]
                )
            
            # Truncate document if too long
            max_text_length = 800000
            truncated_text = self.document_text[:max_text_length]
            if len(self.document_text) > max_text_length:
                truncated_text += "\n[Document truncated due to length...]"
            
            # Create the prompt for query analysis
            prompt = f"""You are a highly analytical and experienced legal document analysis specialist.

Answer the following query based ONLY on the provided document content.

DOCUMENT NAME: {self.document_name}

DOCUMENT CONTENT:
{truncated_text}

QUERY: {query}

Provide your analysis as a JSON object with the following structure:
{{
  "direct_references": [{{"text": "relevant excerpt", "location": "page/section number"}}],
  "interpretation": "string - Legal interpretation of the query based on document",
  "related_principles": ["list of related legal principles"],
  "strategic_insights": ["list of strategic insights for the lawyer"],
  "limitations": ["list of limitations or ambiguities"],
  "recommended_actions": ["list of recommended actions"]
}}

Respond with ONLY the JSON object, no additional text or markdown formatting."""

            # Generate content using Gemini
            start_time = time.time()
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=4096,
                )
            )
            
            # Extract and parse the response
            raw_response = response.text.strip()
            json_str = self._extract_json_from_response(raw_response)
            query_dict = json.loads(json_str)
            
            # Create and return Pydantic model instance
            return DocumentQuery(**query_dict)
            
        except (json.JSONDecodeError, ValidationError) as e:
            error_details = f"Error during query analysis: {type(e).__name__}: {str(e)}"
            return DocumentQuery(
                direct_references=[{"text": "Error occurred during analysis", "location": "N/A"}],
                interpretation=f"Could not properly analyze your query: '{query}'. Error details: {error_details}",
                related_principles=["Error occurred during analysis"],
                strategic_insights=["Response could not be properly formatted"],
                limitations=[f"Technical error in processing the model's response: {type(e).__name__}"],
                recommended_actions=["Try rephrasing your question", "Check the server logs for API errors"]
            )
        except Exception as e:
            return DocumentQuery(
                direct_references=[{"text": "Unknown Error", "location": "N/A"}],
                interpretation=f"An unexpected error occurred during analysis: {str(e)}",
                related_principles=[],
                strategic_insights=[],
                limitations=["Unknown system error"],
                recommended_actions=["Contact support"]
            )


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create a document analyzer instance for each session
document_sessions: Dict[str, LegalDocumentAnalyzer] = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Legal Document API is running"}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
    
    if session_id not in document_sessions:
        try:
            document_sessions[session_id] = LegalDocumentAnalyzer()
        except ValueError as e:
            return jsonify({"error": str(e)}), 500
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_content = file.read()
        
        analyzer = document_sessions[session_id]
        document_text = analyzer.extract_text_from_document(file_content, filename)
        
        if document_text.startswith("Error") or document_text.startswith("Unsupported"):
            return jsonify({"error": document_text}), 400
        
        analyzer.document_text = document_text
        
        return jsonify({
            "message": "File uploaded and text extracted successfully",
            "filename": filename,
            "session_id": session_id,
            "text_length": len(document_text)
        }), 200
    else:
        return jsonify({"error": f"File type not allowed. Supported types are: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
    
    if session_id not in document_sessions:
        return jsonify({"error": "No document found for this session. Please upload a document first."}), 404
    
    analyzer = document_sessions[session_id]
    
    if not analyzer.document_text:
        return jsonify({"error": "No document text found. Please upload a valid document."}), 400
    
    start_time = time.time()
    summary = analyzer.summarize_document()
    processing_time = time.time() - start_time
    
    summary_dict = summary.dict()
    
    response = {
        "summary": summary_dict,
        "processing_time": f"{processing_time:.2f} seconds"
    }
    
    return jsonify(response), 200

@app.route('/api/query', methods=['POST'])
def query_document():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query')
    
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if session_id not in document_sessions:
        return jsonify({"error": "No document found for this session. Please upload a document first."}), 404
    
    analyzer = document_sessions[session_id]
    
    if not analyzer.document_text:
        return jsonify({"error": "No document text found. Please upload a valid document."}), 400
    
    start_time = time.time()
    response = analyzer.query_document(query)
    processing_time = time.time() - start_time
    
    response_dict = response.dict()
    
    result = {
        "response": response_dict,
        "processing_time": f"{processing_time:.2f} seconds"
    }
    
    return jsonify(result), 200

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in document_sessions:
        del document_sessions[session_id]
        return jsonify({"message": f"Session {session_id} deleted successfully"}), 200
    else:
        return jsonify({"error": "Session not found"}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum file size is 16MB"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error(f"Internal Server Error: {error}")
    return jsonify({"error": "Internal server error occurred. Check server logs."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))