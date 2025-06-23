from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, make_response, send_file
import os
import requests
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import io
import csv
import json
from io import StringIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
try:
    from model import AdvancedBERTClassifier
except ImportError:
    print("Warning: AdvancedBERTClassifier not found. Please ensure model.py exists.")

app = Flask(__name__)
app.secret_key = 'bloom-classification-app-2025-secret'  # Use secret_key, not config
app.config['SECRET_KEY'] = 'bloom-classification-app-2025-secret'  # Backup


# Configure upload folder
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Student and results data files
STUDENTS_FILE = os.path.join(UPLOAD_FOLDER, 'students.json')
RESULTS_FILE = os.path.join(UPLOAD_FOLDER, 'results.json')

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_students():
    """Load student data from file"""
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Create default students file
        default_students = {
            "students": [
                {
                    "email": "student1@example.com",
                    "password": "password123",
                    "name": "John Doe"
                },
                {
                    "email": "student2@example.com", 
                    "password": "password123",
                    "name": "Jane Smith"
                }
            ]
        }
        save_students(default_students)
        return default_students

def save_students(students_data):
    """Save student data to file"""
    with open(STUDENTS_FILE, 'w') as f:
        json.dump(students_data, f, indent=2)

def load_results():
    """Load results data from file"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"results": []}

def save_results(results_data):
    """Save results data to file"""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results_data, f, indent=2)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Student login - UPDATED with proper authentication
@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    """Student login page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        students_data = load_students()
        
        # Check credentials
        for student in students_data['students']:
            if student['email'] == email and student['password'] == password:
                session['student_email'] = email
                session['student_name'] = student['name']
                session['student_logged_in'] = True
                flash('Login successful!', 'success')
                return redirect(url_for('student_dashboard'))
        
        flash('Invalid email or password!', 'error')
        return render_template('student/student_login.html', error='Invalid credentials')
    
    return render_template('student/student_login.html')

# Student dashboard - UPDATED
@app.route('/student/dashboard')
def student_dashboard():
    """Student dashboard - requires login"""
    if not session.get('student_logged_in') or 'student_email' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('student_login'))
    
    return render_template('student/student_dashboard.html', 
                         student_name=session.get('student_name'))

# Student logout - UPDATED
@app.route('/student/logout')
def student_logout():
    """Student logout"""
    session.pop('student_logged_in', None)
    session.pop('student_email', None)
    session.pop('student_name', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('home'))

# NEW - API to save student quiz results
@app.route('/api/student/save-results', methods=['POST'])
def save_student_results():
    """Save student quiz results"""
    if not session.get('student_logged_in') or 'student_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        
        result_entry = {
            'email': session['student_email'],
            'name': session['student_name'],
            'score': data.get('score'),
            'total_score': data.get('totalScore'),
            'level': data.get('level'),
            'answers': data.get('answers'),
            'date': datetime.now().isoformat(),
            'timestamp': datetime.now().timestamp()
        }
        
        results_data = load_results()
        
        # Remove any previous results for this student
        results_data['results'] = [r for r in results_data['results'] 
                                 if r['email'] != session['student_email']]
        
        # Add new result
        results_data['results'].append(result_entry)
        
        save_results(results_data)
        
        return jsonify({'success': True, 'message': 'Results saved successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# NEW - API to get student's previous results
@app.route('/api/student/results')
def get_student_results():
    """Get student's previous results"""
    if not session.get('student_logged_in') or 'student_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    try:
        results_data = load_results()
        
        # Find this student's results
        student_results = [r for r in results_data['results'] 
                         if r['email'] == session['student_email']]
        
        if student_results:
            # Return the most recent result
            latest_result = max(student_results, key=lambda x: x['timestamp'])
            return jsonify({
                'success': True,
                'hasResults': True,
                'data': {
                    'email': latest_result['email'],
                    'score': latest_result['score'],
                    'level': latest_result['level'],
                    'date': latest_result['date'][:10],  # Just the date part
                    'tests_taken': len(student_results)
                }
            })
        else:
            return jsonify({
                'success': True,
                'hasResults': False,
                'message': 'No previous results found'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/student/download-pdf')
def download_student_pdf():
    """Download student's results as PDF"""
    if not session.get('student_logged_in') or 'student_email' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('student_login'))
    
    try:
        import pdfkit
        from flask import make_response
        
        # Get student's latest results
        results_data = load_results()
        student_results = [r for r in results_data['results'] 
                         if r['email'] == session['student_email']]
        
        if not student_results:
            flash('No results found. Please take the assessment first.', 'error')
            return redirect(url_for('student_dashboard'))
        
        # Get the latest result
        latest_result = max(student_results, key=lambda x: x['timestamp'])
        
        # Generate descriptions and recommendations based on score
        score = latest_result['score']
        # (Add the same logic from your JavaScript for level/description/recommendations)
        
        # Create HTML for PDF
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>AQ Assessment Results</title>
            <style>
                /* Add your CSS styling here */
            </style>
        </head>
        <body>
            <!-- Your results HTML structure here -->
        </body>
        </html>
        """
        
        # Generate PDF
        pdf = pdfkit.from_string(html_content, False)
        
        # Create response
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=AQ_Results_{session["student_name"]}.pdf'
        
        return response
        
    except Exception as e:
        flash(f'Error generating PDF: {str(e)}', 'error')
        return redirect(url_for('student_dashboard'))

# Teacher login page - UPDATED to use hardcoded credentials that match paste.txt
@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Use same credentials as in paste.txt
        if email == 'teacher@example.com' and password == 'teacher123':
            session['teacher_logged_in'] = True
            session['teacher_email'] = email
            flash('Login successful!', 'success')
            return redirect(url_for('teacher_dashboard'))
        else:
            flash('Invalid email or password!', 'error')
    
    return render_template('teacher/teacher_login.html')

# Teacher dashboard - UPDATED to show all student results
@app.route('/teacher/dashboard')
def teacher_dashboard():
    if not session.get('teacher_logged_in'):
        flash('Please login first!', 'error')
        return redirect(url_for('teacher_login'))
    
    # Load all results for teacher view
    results_data = load_results()
    
    return render_template('teacher/teacher_dashboard.html', 
                         results=results_data['results'])

# Teacher logout
@app.route('/teacher/logout')
def teacher_logout():
    session.pop('teacher_logged_in', None)
    session.pop('teacher_email', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('home'))

# Global variable to store the predictor instance
_predictor = None

def get_predictor():
    """Get or create the predictor instance (lazy loading)"""
    global _predictor
    if _predictor is None:
        try:
            from predict import BloomPredictor
            print("Loading BERT model for first-time use...")
            _predictor = BloomPredictor(model_path='best_model.pth')
            print("Model loaded successfully!")
        except ImportError:
            print("Warning: BloomPredictor not found. Using mock predictor.")
            _predictor = MockPredictor()
    return _predictor

class MockPredictor:
    """Mock predictor for testing when actual model is not available"""
    def predict_from_file(self, file_path, question_column='question', sheet_name=0):
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Create mock predictions
        import random
        mock_results = []
        for idx, row in df.iterrows():
            question = row.get(question_column, f"Question {idx+1}")
            level = random.randint(1, 6)
            descriptions = {
                1: "Remember - Recall facts and basic concepts",
                2: "Understand - Explain ideas or concepts", 
                3: "Apply - Use information in new situations",
                4: "Analyze - Draw connections among ideas",
                5: "Evaluate - Justify a stand or decision",
                6: "Create - Produce new or original work"
            }
            mock_results.append({
                'question': question,
                'predicted_level': level,
                'predicted_description': descriptions[level],
                'confidence': round(random.uniform(0.7, 0.95), 3),
                'prediction_error': None
            })
        
        return pd.DataFrame(mock_results)

def classify_questions(filepath):
    """
    Flask integration function to classify questions from uploaded file
    
    Args:
        filepath (str): Path to the uploaded CSV or Excel file
        
    Returns:
        list: List of dictionaries containing question, bloom_level, confidence
    """
    try:
        # Get the predictor instance (loads model only when needed)
        predictor = get_predictor()
        
        # Process the file
        results_df = predictor.predict_from_file(
            file_path=filepath,
            question_column='question',
            sheet_name=0  # First sheet for Excel files
        )
        
        # Convert to the format expected by Flask
        flask_results = []
        
        for index, row in results_df.iterrows():
            # Skip rows with prediction errors
            if pd.notna(row.get('prediction_error')) and row['prediction_error']:
                continue
                
            result_dict = {
                'question': str(row['question']) if pd.notna(row['question']) else 'N/A',
                'bloom_level': f"L{int(row['predicted_level'])}" if pd.notna(row['predicted_level']) else 'Unknown',
                'bloom_description': str(row['predicted_description']) if pd.notna(row['predicted_description']) else 'Unknown',
                'confidence': f"{float(row['confidence'])*100:.1f}%" if pd.notna(row['confidence']) else '0.0%'
            }
            flask_results.append(result_dict)
        
        return flask_results
        
    except Exception as e:
        # Return error information that Flask can handle
        return [{'error': f'Classification failed: {str(e)}'}]

def create_results_table_html(results):
    """Create HTML table from results for template rendering"""
    if not results or (len(results) == 1 and 'error' in results[0]):
        return "No results available"
    
    html = """
    <table class="results-table">
        <thead>
            <tr>
                <th>Question</th>
                <th>Bloom Level</th>
                <th>Description</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for result in results:
        if 'error' not in result:
            html += f"""
            <tr>
                <td class="question-cell">{result['question']}</td>
                <td class="level-cell">{result['bloom_level']}</td>
                <td class="description-cell">{result['bloom_description']}</td>
                <td class="confidence-cell">{result['confidence']}</td>
            </tr>
            """
    
    html += """
        </tbody>
    </table>
    """
    
    return html

# File upload route for teacher dashboard - UPDATED VERSION
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload from teacher dashboard and run BERT classification"""
    # Check if user is logged in
    if not session.get('teacher_logged_in'):
        flash('Please login first!', 'error')
        return redirect(url_for('teacher_login'))
    
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('teacher_dashboard'))
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('teacher_dashboard'))
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a CSV or Excel file.', 'error')
            return redirect(url_for('teacher_dashboard'))
        
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"File saved: {filepath}")  # Debug
            flash(f'File "{filename}" uploaded successfully! Processing...', 'info')
            
            # INTEGRATE WITH ML MODEL HERE
            try:
                # Use the local classify_questions function
                results = classify_questions(filepath)
                print(f"Classification completed - Results count: {len(results)}")  # Debug
                
                if results and len(results) > 0:
                    if 'error' in results[0]:
                        flash(f'Classification error: {results[0]["error"]}', 'error')
                    else:
                        flash('Classification completed successfully!', 'success')
                        print(f"First result sample: {results[0]}")  # Debug
                else:
                    flash('No results generated from classification', 'warning')
                    results = []
                
                # Store results in session to pass to results page
                session['classification_results'] = results
                session['uploaded_filename'] = filename
                
                return redirect(url_for('results'))
                
            except Exception as e:
                print(f"Classification error: {str(e)}")  # Debug
                flash(f'Error during classification: {str(e)}', 'error')
                return redirect(url_for('teacher_dashboard'))
            
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Debug
        flash(f'Error uploading file: {str(e)}', 'error')
        return redirect(url_for('teacher_dashboard'))
        
@app.route('/results')
def results():
    """Display classification results using the result.html template"""
    if not session.get('teacher_logged_in'):
        flash('Please login first!', 'error')
        return redirect(url_for('teacher_login'))
    
    # Get results from session
    results = session.get('classification_results', [])
    filename = session.get('uploaded_filename', None)
    
    # Debug logging
    app.logger.info(f"Results route called - Results count: {len(results)}")
    app.logger.info(f"Filename: {filename}")
    
    # Validate and clean results
    valid_results = []
    error_count = 0
    
    for result in results:
        if isinstance(result, dict) and 'error' not in result:
            # Validate required fields
            required_fields = ['question', 'bloom_level', 'bloom_description', 'confidence']
            if all(field in result for field in required_fields):
                valid_results.append(result)
            else:
                error_count += 1
                app.logger.warning(f"Invalid result structure: {result}")
        elif isinstance(result, dict) and 'error' in result:
            error_count += 1
            app.logger.error(f"Error in result: {result['error']}")
    
    # Log statistics
    app.logger.info(f"Valid results: {len(valid_results)}, Errors: {error_count}")
    
    # Update session with cleaned results
    session['classification_results'] = valid_results
    
    return render_template('result.html',
                         results=valid_results,
                         filename=filename,
                         total_count=len(results),
                         valid_count=len(valid_results),
                         error_count=error_count)

@app.route('/download-pdf')
def download_pdf():
    """Generate and download PDF of classification results"""
    if not session.get('teacher_logged_in'):
        flash('Please login first!', 'error')
        return redirect(url_for('teacher_login'))
    
    results = session.get('classification_results', [])
    filename = session.get('uploaded_filename', 'Unknown')
    
    if not results:
        flash('No results to download!', 'error')
        return redirect(url_for('results'))
    
    try:
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        elements = []
        
        # Enhanced styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        
        subtitle_style = ParagraphStyle(
            'SubTitle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=1,
            textColor=colors.grey
        )
        
        # Title and metadata
        title = Paragraph(f"BERT Classification Results", title_style)
        elements.append(title)
        
        subtitle = Paragraph(f"File: {filename} | Total Questions: {len(results)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style)
        elements.append(subtitle)
        elements.append(Spacer(1, 20))
        
        # Prepare table data with better formatting
        table_data = [['#', 'Question', 'Bloom Level', 'Description', 'Confidence']]
        
        for i, result in enumerate(results, 1):
            if 'error' not in result:
                # Smart truncation for questions
                question = result['question']
                if len(question) > 60:
                    question = question[:57] + "..."
                
                # Format confidence as percentage
                confidence = result.get('confidence', 'N/A')
                if isinstance(confidence, (int, float)):
                    confidence = f"{confidence:.2%}"
                elif isinstance(confidence, str) and confidence.replace('.', '').isdigit():
                    confidence = f"{float(confidence):.2%}"
                
                description = result['bloom_description']
                if len(description) > 40:
                    description = description[:37] + "..."
                
                table_data.append([
                    str(i),
                    Paragraph(question, styles['Normal']),
                    result['bloom_level'],
                    Paragraph(description, styles['Normal']),
                    confidence
                ])
        
        # Create table with improved styling
        table = Table(table_data, colWidths=[0.4*inch, 2.8*inch, 1*inch, 2*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Center align row numbers
            ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),  # Center align confidence
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        elements.append(table)
        
        # Add footer with statistics
        elements.append(Spacer(1, 30))
        
        # Calculate Bloom level distribution
        bloom_counts = {}
        for result in results:
            if 'error' not in result:
                level = result.get('bloom_level', 'Unknown')
                bloom_counts[level] = bloom_counts.get(level, 0) + 1
        
        # Add distribution info
        if bloom_counts:
            elements.append(Paragraph("Bloom's Taxonomy Distribution:", styles['Heading2']))
            for level, count in sorted(bloom_counts.items()):
                elements.append(Paragraph(f"• {level}: {count} questions", styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        # Create response
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=bloom_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        # Log successful download
        app.logger.info(f"PDF generated successfully for {len(results)} results")
        flash('PDF downloaded successfully!', 'success')
        
        return response
        
    except Exception as e:
        app.logger.error(f'Error generating PDF: {str(e)}', exc_info=True)
        flash(f'Error generating PDF: {str(e)}', 'error')
        return redirect(url_for('results'))

@app.route('/export-csv')
def export_csv():
    """Export classification results as CSV with enhanced formatting"""
    if not session.get('teacher_logged_in'):
        flash('Please login first!', 'error')
        return redirect(url_for('teacher_login'))
    
    results = session.get('classification_results', [])
    filename = session.get('uploaded_filename', 'Unknown')
    
    if not results:
        flash('No results to export!', 'error')
        return redirect(url_for('results'))
    
    try:
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        
        # Write metadata header
        writer.writerow(['# BERT Classification Results'])
        writer.writerow([f'# Source File: {filename}'])
        writer.writerow([f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([f'# Total Questions: {len(results)}'])
        writer.writerow([])  # Empty row
        
        # Write column headers
        writer.writerow(['Question Number', 'Question Text', 'Bloom Level', 'Description', 'Confidence Score'])
        
        # Write data with proper formatting
        for i, result in enumerate(results, 1):
            if 'error' not in result:
                # Format confidence
                confidence = result.get('confidence', 'N/A')
                if isinstance(confidence, (int, float)):
                    confidence = f"{confidence:.4f}"
                
                writer.writerow([
                    i,
                    result['question'].strip(),
                    result['bloom_level'],
                    result['bloom_description'].strip(),
                    confidence
                ])
        
        # Add summary statistics
        writer.writerow([])  # Empty row
        writer.writerow(['# Summary Statistics'])
        
        # Calculate Bloom level distribution
        bloom_counts = {}
        for result in results:
            if 'error' not in result:
                level = result.get('bloom_level', 'Unknown')
                bloom_counts[level] = bloom_counts.get(level, 0) + 1
        
        for level, count in sorted(bloom_counts.items()):
            writer.writerow([f'# {level}', count])
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=bloom_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # Log successful export
        app.logger.info(f"CSV exported successfully for {len(results)} results")
        flash('CSV exported successfully!', 'success')
        
        return response
        
    except Exception as e:
        app.logger.error(f'Error generating CSV: {str(e)}', exc_info=True)
        flash(f'Error generating CSV: {str(e)}', 'error')
        return redirect(url_for('results'))

# Additional utility route for results statistics
@app.route('/results-stats')
def results_stats():
    """Get statistics about classification results (AJAX endpoint)"""
    if not session.get('teacher_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    results = session.get('classification_results', [])
    
    if not results:
        return jsonify({'error': 'No results found'}), 404
    
    # Calculate statistics
    bloom_counts = {}
    confidence_scores = []
    
    for result in results:
        if 'error' not in result:
            level = result.get('bloom_level', 'Unknown')
            bloom_counts[level] = bloom_counts.get(level, 0) + 1
            
            confidence = result.get('confidence')
            if isinstance(confidence, (int, float)):
                confidence_scores.append(confidence)
            elif isinstance(confidence, str) and confidence.replace('.', '').isdigit():
                confidence_scores.append(float(confidence))
    
    # Calculate confidence statistics
    conf_stats = {}
    if confidence_scores:
        conf_stats = {
            'mean': sum(confidence_scores) / len(confidence_scores),
            'min': min(confidence_scores),
            'max': max(confidence_scores),
            'count': len(confidence_scores)
        }
    
    return jsonify({
        'total_results': len(results),
        'bloom_distribution': bloom_counts,
        'confidence_stats': conf_stats
    })

# API endpoint for getting results as JSON (for JavaScript)
@app.route('/api/results')
def api_results():
    """API endpoint to get classification results as JSON"""
    if not session.get('teacher_logged_in'):
        return jsonify({'error': 'Not authenticated'}), 401
    
    results = session.get('classification_results', [])
    return jsonify({
        'results': results,
        'filename': session.get('uploaded_filename', 'Unknown'),
        'total_count': len(results)
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/138Dy7Ecwqeskm7D0jJY9vIpNJW_uBHM3jSiYS5dOmiE/export?format=csv"

# Fetch data from Google Sheets
def fetch_google_sheets_data():
    try:
        response = requests.get(GOOGLE_SHEET_URL)
        response.raise_for_status()
        
        # Parse CSV data
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        return df
    except Exception as e:
        print(f"Error fetching Google Sheets data: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)