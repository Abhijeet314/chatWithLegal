from flask import Flask, request, jsonify
import together
import json
import re
import os
import time
import tempfile
import PyPDF2
import docx
from io import StringIO
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Pydantic models for structured output
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

class LegalDocumentAnalyzer:
    def __init__(self):
        # Initialize Together AI client
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        if not self.together_api_key:
            raise ValueError("API key not found. Please set the TOGETHER_API_KEY environment variable.")
            
        # Initialize the Together client
        self.client = together.Together(api_key=self.together_api_key)
        self.document_text = ""
        self.document_name = ""
        
    def extract_text_from_document(self, file_path, file_name) -> str:
        """Extract text from various document formats"""
        try:
            file_extension = file_name.split('.')[-1].lower()
            self.document_name = file_name
            
            if file_extension == 'pdf':
                # Extract text from PDF
                text = ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text() + "\n\n"
                return text
                
            elif file_extension == 'docx':
                # Process DOCX files
                doc = docx.Document(file_path)
                text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
                
            elif file_extension == 'txt':
                # Process text files
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                return text
                
            else:
                return f"Unsupported file format: .{file_extension}"
                
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def summarize_document(self) -> DocumentSummary:
        """Generate a structured summary of the uploaded legal document"""
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
            
            # Truncate document if too long (context window consideration)
            max_text_length = 100000  # Adjust based on model capabilities
            truncated_text = self.document_text[:max_text_length]
            if len(self.document_text) > max_text_length:
                truncated_text += "\n[Document truncated due to length...]"
            
            # Prepare the prompt
            prompt = f"""You are an expert legal AI assistant specializing in analyzing legal documents. You have been provided with the following legal document:

DOCUMENT NAME: {self.document_name}

DOCUMENT CONTENT:
{truncated_text}

Analyze this document comprehensively as a legal professional would. Provide a detailed structured analysis covering:

1. The specific type of legal document (e.g., contract, pleading, judgment, etc.)
2. All parties mentioned in the document
3. Important dates mentioned with their context
4. Primary legal subjects covered
5. Key legal provisions or clauses
6. Critical obligations mentioned
7. Potential legal issues or ambiguities identified
8. A concise but comprehensive summary of the document's purpose and content

Your response MUST be in valid JSON format that matches this structure exactly:
{{
"document_type": "Type of legal document",
"parties_involved": ["Party 1", "Party 2", ...],
"key_dates": [
    {{"date": "YYYY-MM-DD or description", "context": "What this date refers to"}},
    ...
],
"primary_subjects": ["Subject 1", "Subject 2", ...],
"key_provisions": ["Provision 1", "Provision 2", ...],
"critical_obligations": ["Obligation 1", "Obligation 2", ...],
"potential_issues": ["Issue 1", "Issue 2", ...],
"summary": "Comprehensive summary of the document"
}}

IMPORTANT: The JSON response must be valid JSON with no leading or trailing whitespace around keys. Include at least 3-5 items per field where appropriate. Do not include any text outside of the JSON structure. The entire response should be valid JSON."""
            
            # Get the response from the model
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-classifier",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=4096
            )
            
            response_time = time.time() - start_time
            
            # Parse JSON response
            raw_response = response.choices[0].message.content
            clean_text = self._clean_json_response(raw_response)
            analysis_dict = json.loads(clean_text)
            
            # Create and return Pydantic model instance
            return DocumentSummary(**analysis_dict)
            
        except Exception as e:
            return DocumentSummary(
                document_type="Error",
                parties_involved=[],
                key_dates=[],
                primary_subjects=[],
                key_provisions=[],
                critical_obligations=[],
                potential_issues=[],
                summary=f"Error analyzing document: {str(e)}"
            )
    
    def query_document(self, query: str) -> DocumentQuery:
        """Generate a response to a specific query about the document with improved handling"""
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
            
            # Truncate document if too long (context window consideration)
            max_text_length = 100000
            truncated_text = self.document_text[:max_text_length]
            if len(self.document_text) > max_text_length:
                truncated_text += "\n[Document truncated due to length...]"
            
            # Simplified prompt to reduce JSON parsing issues
            prompt = f"""You are an expert legal AI assistant. Analyze this document in relation to the query.

    DOCUMENT NAME: {self.document_name}

    DOCUMENT CONTENT:
    {truncated_text}

    QUERY: {query}

    Provide a comprehensive analysis focused specifically on answering this query.
    Focus on finding DIFFERENT and RELEVANT information for THIS SPECIFIC QUERY.
    Do not provide generic information about the document that doesn't relate to the query.

    Your response must be JSON with this structure:
    {{
    "direct_references": [
        {{"text": "Exact relevant text from document", "location": "Section information"}},
        {{"text": "Another relevant excerpt", "location": "Section information"}}
    ],
    "interpretation": "Your specific interpretation addressing THIS query",
    "related_principles": ["Principle 1", "Principle 2"],
    "strategic_insights": ["Insight 1", "Insight 2"],
    "limitations": ["Limitation 1", "Limitation 2"],
    "recommended_actions": ["Action 1", "Action 2"]
    }}

    IMPORTANT:
    1. Output ONLY valid JSON - no markdown, no code blocks
    2. Every section MUST contain information SPECIFIC to this query
    3. Do not repeat the same information across different queries
    4. All JSON properties must be present even if arrays are empty ([])
    """
            
            # Get response with increased temperature for variety
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-classifier",
                messages=[
                    {"role": "system", "content": "You are a legal document analysis specialist. Provide detailed and VARIED responses to different queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,  # Increased for more variation
                max_tokens=4096
            )
            
            # Store raw response for debugging
            raw_response = response.choices[0].message.content
            
            # Better JSON handling
            try:
                # First attempt: direct parsing
                query_dict = json.loads(raw_response)
            except json.JSONDecodeError:
                # Second attempt: try to extract JSON from markdown
                if "```json" in raw_response:
                    match = re.search(r"```json\n(.*?)\n```", raw_response, re.DOTALL)
                    if match:
                        try:
                            query_dict = json.loads(match.group(1).strip())
                        except:
                            raise ValueError("Failed to parse JSON from code block")
                    else:
                        raise ValueError("Could not extract JSON from markdown")
                elif "```" in raw_response:
                    match = re.search(r"```\n(.*?)\n```", raw_response, re.DOTALL)
                    if match:
                        try:
                            query_dict = json.loads(match.group(1).strip())
                        except:
                            raise ValueError("Failed to parse JSON from code block")
                    else:
                        raise ValueError("Could not extract JSON from markdown")
                else:
                    # Last attempt: fix common JSON issues
                    fixed_text = raw_response.replace("'", "\"")
                    fixed_text = re.sub(r',\s*}', '}', fixed_text)
                    fixed_text = re.sub(r',\s*]', ']', fixed_text)
                    try:
                        query_dict = json.loads(fixed_text)
                    except:
                        raise ValueError("All JSON parsing attempts failed")
            
            # Create and return Pydantic model instance
            return DocumentQuery(**query_dict)
            
        except Exception as e:
            # More informative response instead of default template
            return DocumentQuery(
                direct_references=[{"text": "Error occurred during analysis", "location": "N/A"}],
                interpretation=f"Could not properly analyze your query: '{query}'. Please try a different question or format.",
                related_principles=["Error occurred during analysis"],
                strategic_insights=["Error occurred during analysis"],
                limitations=[f"Error: {str(e)}", "Try a more specific question"],
                recommended_actions=["Try rephrasing your question", "Check if your question is specific to the document content"]
            )
            
    def _clean_json_response(self, raw_response: str) -> str:
        """Simpler and more robust function to clean and repair LLM JSON responses"""
        # Remove any whitespace at beginning and end
        clean_text = raw_response.strip()
        
        # First try: direct parsing
        try:
            json.loads(clean_text)
            return clean_text
        except json.JSONDecodeError:
            pass
        
        # Second try: handle markdown code blocks
        if "```json" in clean_text:
            try:
                match = re.search(r"```json\n(.*?)\n```", clean_text, re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    json.loads(extracted)
                    return extracted
            except:
                pass
        elif "```" in clean_text:
            try:
                match = re.search(r"```\n(.*?)\n```", clean_text, re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    json.loads(extracted)
                    return extracted
            except:
                pass
        
        # Third try: basic JSON repairs
        try:
            # Replace single quotes with double quotes
            fixed_text = clean_text.replace("'", "\"")
            # Remove trailing commas
            fixed_text = re.sub(r',\s*}', '}', fixed_text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            json.loads(fixed_text)
            return fixed_text
        except:
            pass
        
        # If we got here, all parsing attempts failed
        # Create a default response
        default_response = {
            "direct_references": [
                {"text": "Error parsing the model's response", "location": "N/A"}
            ],
            "interpretation": "Error parsing model response",
            "related_principles": ["Unable to extract information due to technical error"],
            "strategic_insights": ["Response could not be properly formatted"],
            "limitations": ["Technical error in processing the model's response"],
            "recommended_actions": ["Try phrasing your question differently", "Contact technical support if the problem persists"]
        }
        
        return json.dumps(default_response)

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create a document analyzer instance for each session
document_sessions = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Legal Document API is running"}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Check if file part exists in request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check session ID
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
    
    # Create a new analyzer for this session if needed
    if session_id not in document_sessions:
        try:
            document_sessions[session_id] = LegalDocumentAnalyzer()
        except ValueError as e:
            return jsonify({"error": str(e)}), 500
    
    # Process the file if it has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        # Extract text from document
        analyzer = document_sessions[session_id]
        document_text = analyzer.extract_text_from_document(file_path, filename)
        
        if document_text.startswith("Error") or document_text.startswith("Unsupported"):
            return jsonify({"error": document_text}), 400
        
        # Store document text in analyzer
        analyzer.document_text = document_text
        
        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "session_id": session_id,
            "text_length": len(document_text)
        }), 200
    else:
        return jsonify({"error": f"File type not allowed. Supported types are: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    # Get session ID from request
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
    
    # Check if session exists
    if session_id not in document_sessions:
        return jsonify({"error": "No document found for this session. Please upload a document first."}), 404
    
    analyzer = document_sessions[session_id]
    
    # Check if document text exists
    if not analyzer.document_text:
        return jsonify({"error": "No document text found. Please upload a valid document."}), 400
    
    # Generate summary
    start_time = time.time()
    summary = analyzer.summarize_document()
    processing_time = time.time() - start_time
    
    # Convert Pydantic model to dictionary
    summary_dict = summary.dict()
    
    # Add processing time info
    response = {
        "summary": summary_dict,
        "processing_time": f"{processing_time:.2f} seconds"
    }
    
    return jsonify(response), 200

@app.route('/api/query', methods=['POST'])
def query_document():
    # Get data from request
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query')
    
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Check if session exists
    if session_id not in document_sessions:
        return jsonify({"error": "No document found for this session. Please upload a document first."}), 404
    
    analyzer = document_sessions[session_id]
    
    # Check if document text exists
    if not analyzer.document_text:
        return jsonify({"error": "No document text found. Please upload a valid document."}), 400
    
    # Process query
    start_time = time.time()
    response = analyzer.query_document(query)
    processing_time = time.time() - start_time
    
    # Convert Pydantic model to dictionary
    response_dict = response.dict()
    
    # Add processing time info
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

# Add error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum file size is 16MB"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))