from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2
import docx
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
print(f"API Key Status: {'Found' if GEMINI_API_KEY else 'NOT FOUND'}")

model = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try different model names
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro'
        ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                test_response = model.generate_content("Say 'ready'")
                print(f"✓ Gemini model '{model_name}' initialized successfully")
                break
            except Exception as model_error:
                print(f"✗ Model '{model_name}' failed: {str(model_error)}")
                continue
                
    except Exception as e:
        print(f"✗ Error initializing Gemini: {str(e)}")
        model = None
else:
    print("✗ GEMINI_API_KEY not found")

# In-memory session storage (use Redis or DB in production)
sessions = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        return None

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error extracting DOCX text: {str(e)}")
        return None

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error extracting TXT text: {str(e)}")
        return None

def extract_text(file_path, filename):
    """Extract text based on file type"""
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['docx', 'doc']:
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    else:
        return None

def analyze_document_with_ai(text):
    """Analyze document using Gemini AI"""
    if not model:
        return generate_sample_summary(text)
    
    try:
        prompt = f"""You are an expert legal document analyzer. Analyze the following legal document and provide a comprehensive summary.

Document Text:
{text[:8000]}  

Provide a detailed analysis in the following JSON format:
{{
  "document_type": "Type of legal document (e.g., Contract, Agreement, Legal Notice, etc.)",
  "key_parties": ["List of parties involved"],
  "main_clauses": ["List of 5-7 main clauses or sections"],
  "important_dates": ["List of important dates mentioned"],
  "obligations": ["List of key obligations for each party"],
  "rights": ["List of key rights granted"],
  "risks": ["List of 3-5 potential risks or concerns"],
  "recommendations": ["List of 3-5 recommendations or action items"],
  "summary": "A comprehensive 2-3 paragraph summary of the document",
  "legal_terms": ["List of important legal terms used"]
}}

Ensure the response is valid JSON and all fields are populated with relevant information from the document."""

        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.8,
                'max_output_tokens': 2048,
            }
        )
        
        # Parse JSON response
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.text
        
        summary = json.loads(json_str)
        summary["source"] = "AI Generated"
        return summary
        
    except Exception as e:
        print(f"AI analysis error: {str(e)}")
        return generate_sample_summary(text)

def generate_sample_summary(text):
    """Generate a basic summary when AI is unavailable"""
    words = text.split()
    word_count = len(words)
    
    return {
        "document_type": "Legal Document",
        "key_parties": ["Party information not available without AI analysis"],
        "main_clauses": [
            "Document requires AI analysis for detailed clause identification",
            f"Document contains approximately {word_count} words",
            "Multiple sections and provisions present"
        ],
        "important_dates": ["Date extraction requires AI analysis"],
        "obligations": ["Detailed obligation analysis requires AI processing"],
        "rights": ["Rights analysis requires AI processing"],
        "risks": [
            "Complete document review recommended",
            "Legal consultation advised for important decisions",
            "Verify all information with qualified legal professional"
        ],
        "recommendations": [
            "Upload document for full AI analysis",
            "Consult with a legal professional",
            "Review all clauses carefully"
        ],
        "summary": f"This legal document contains approximately {word_count} words across multiple sections. A complete AI-powered analysis requires a valid Gemini API key. The document should be reviewed by a qualified legal professional for accurate interpretation.",
        "legal_terms": ["Analysis requires AI processing"],
        "source": "Basic Analysis (No AI)"
    }

def answer_query_with_ai(text, query, context):
    """Answer query about document using AI"""
    if not model:
        return {
            "answer": "AI query processing is not available. Please ensure Gemini API key is configured.",
            "relevant_sections": [],
            "confidence": "low",
            "source": "No AI Available"
        }
    
    try:
        prompt = f"""You are a legal document assistant. Answer the following question about the legal document based on the provided text.

Document Text:
{text[:6000]}

Previous Context (if any):
{json.dumps(context[-3:]) if context else "No previous context"}

User Question: {query}

Provide your response in the following JSON format:
{{
  "answer": "Detailed answer to the question based on the document",
  "relevant_sections": ["Quotes or references to relevant sections from the document"],
  "confidence": "high/medium/low - your confidence in the answer",
  "additional_notes": "Any additional relevant information or caveats"
}}

Be precise, cite specific sections when possible, and indicate if information is not found in the document."""

        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.4,
                'top_p': 0.9,
                'max_output_tokens': 1024,
            }
        )
        
        # Parse JSON response
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.text
        
        result = json.loads(json_str)
        result["source"] = "AI Generated"
        return result
        
    except Exception as e:
        print(f"Query processing error: {str(e)}")
        return {
            "answer": f"I encountered an error processing your query: {str(e)}",
            "relevant_sections": [],
            "confidence": "low",
            "source": "Error"
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_available": model is not None,
        "api_key_configured": GEMINI_API_KEY is not None,
        "active_sessions": len(sessions)
    }), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process a legal document"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{session_id}_{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Extract text
        print(f"Extracting text from {filename}...")
        text = extract_text(file_path, filename)
        
        if not text:
            os.remove(file_path)
            return jsonify({"error": "Failed to extract text from document"}), 400
        
        # Store session data
        sessions[session_id] = {
            "filename": filename,
            "file_path": file_path,
            "text": text,
            "upload_time": datetime.now().isoformat(),
            "query_history": []
        }
        
        print(f"✓ File uploaded successfully: {filename} (Session: {session_id})")
        
        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "session_id": session_id,
            "text_length": len(text)
        }), 200
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """Analyze uploaded document"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({"error": "Invalid or expired session"}), 400
        
        session = sessions[session_id]
        text = session['text']
        
        print(f"Analyzing document for session {session_id}...")
        
        # Analyze with AI
        summary = analyze_document_with_ai(text)
        
        # Store summary in session
        session['summary'] = summary
        session['analysis_time'] = datetime.now().isoformat()
        
        print(f"✓ Analysis complete (Source: {summary.get('source', 'Unknown')})")
        
        return jsonify({
            "message": "Document analyzed successfully",
            "summary": summary
        }), 200
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_document():
    """Answer questions about the document"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        query = data.get('query', '').strip()
        
        if not session_id or session_id not in sessions:
            return jsonify({"error": "Invalid or expired session"}), 400
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        session = sessions[session_id]
        text = session['text']
        query_history = session.get('query_history', [])
        
        print(f"Processing query for session {session_id}: {query[:50]}...")
        
        # Get AI response
        response = answer_query_with_ai(text, query, query_history)
        
        # Store in history
        query_record = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        query_history.append(query_record)
        session['query_history'] = query_history
        
        print(f"✓ Query processed (Confidence: {response.get('confidence', 'unknown')})")
        
        return jsonify({
            "message": "Query processed successfully",
            "response": response
        }), 200
        
    except Exception as e:
        print(f"Query error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete session and cleanup files"""
    try:
        if session_id in sessions:
            session = sessions[session_id]
            
            # Delete file if exists
            if 'file_path' in session and os.path.exists(session['file_path']):
                os.remove(session['file_path'])
            
            # Remove from sessions
            del sessions[session_id]
            
            print(f"✓ Session deleted: {session_id}")
            
            return jsonify({"message": "Session deleted successfully"}), 200
        else:
            return jsonify({"message": "Session not found"}), 404
            
    except Exception as e:
        print(f"Delete session error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """Get query history for a session"""
    try:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        
        session = sessions[session_id]
        return jsonify({
            "filename": session.get('filename'),
            "upload_time": session.get('upload_time'),
            "query_history": session.get('query_history', [])
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"Legal Document Analyzer API")
    print(f"Port: {port}")
    print(f"Model Status: {'✓ Ready' if model else '✗ Not Available'}")
    print(f"API Key: {'✓ Configured' if GEMINI_API_KEY else '✗ Not Configured'}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=True)