import os
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename

from src.detector import UnifiedThreatDetector
from src.utils.data_processor import process_profile_url
from src.utils.visualization import generate_report

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_dev_key_change_in_production")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detector
detector = UnifiedThreatDetector()

# Warm up NLP classifier during server startup so first live query is instantaneous
try:
    from src.features.nlp_classifier import get_nlp_classifier
    _nlp_clf = get_nlp_classifier()
    _nlp_clf.classify_text("System warmup test")
    logger.info("DistilBERT NLP engine prewarmed and ready in memory")
except Exception as _warm_err:
    logger.info(f"NLP warmup note: {_warm_err}")

# Server-side stores to prevent cookie size overflow (>4KB)
BATCH_RESULTS_STORE = {}
SINGLE_RESULTS_STORE = {}

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_profile():
    """Process a profile URL or username and display results."""
    if request.method == 'POST':
        profile_url = request.form.get('profile_url')
        platform = request.form.get('platform')
        
        if not profile_url:
            flash('Please enter a profile URL or username', 'error')
            return redirect(url_for('index'))
        
        try:
            # Process the profile data
            profile_data = process_profile_url(profile_url, platform)
            
            # Run the detection
            result = detector.analyze_profile(profile_data)
            
            # Generate visualization for the report
            report_data = generate_report(result, profile_data)
            
            # Store result in server-side store
            analysis_id = str(uuid.uuid4())
            SINGLE_RESULTS_STORE[analysis_id] = {
                'report_data': report_data,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            session['analysis_id'] = analysis_id
            
            return redirect(url_for('results'))
        
        except Exception as e:
            logger.error(f"Error analyzing profile: {str(e)}", exc_info=True)
            flash(f'Error analyzing profile: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for profile analysis."""
    try:
        data = request.get_json()
        
        if not data or 'profile_url' not in data:
            return jsonify({'error': 'Missing profile_url'}), 400
        
        profile_url = data['profile_url']
        platform = data.get('platform', 'twitter')
        
        # Process the profile data
        profile_data = process_profile_url(profile_url, platform)
        
        # Run the detection
        result = detector.analyze_profile(profile_data)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['GET', 'POST'])
def batch_analysis():
    """Handle batch processing of multiple profiles."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                platform = request.form.get('platform', 'twitter')
                
                # Process the batch file
                results = detector.batch_analyze_from_file(filepath, platform)
                
                # Clean up uploaded temp file to conserve disk space
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception:
                    pass
                
                # Store batch results in server-side memory store with LRU pruning (max 5 batches)
                while len(BATCH_RESULTS_STORE) >= 5:
                    oldest_k = next(iter(BATCH_RESULTS_STORE))
                    del BATCH_RESULTS_STORE[oldest_k]
                    
                batch_id = str(uuid.uuid4())
                BATCH_RESULTS_STORE[batch_id] = {
                    'results': results,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                session['batch_id'] = batch_id
                
                return redirect(url_for('batch_results'))
            
            except Exception as e:
                logger.error(f"Error in batch analysis: {str(e)}", exc_info=True)
                flash(f'Error processing batch file: {str(e)}', 'error')
                return redirect(url_for('batch_analysis'))
    
    return render_template('batch.html')

@app.route('/results')
def results():
    """Display the results of a single profile analysis."""
    analysis_id = session.get('analysis_id')
    stored = SINGLE_RESULTS_STORE.get(analysis_id) if analysis_id else None
    
    if not stored and SINGLE_RESULTS_STORE:
        stored = list(SINGLE_RESULTS_STORE.values())[-1]
    
    if not stored:
        flash('No analysis data found. Please analyze a profile first.', 'error')
        return redirect(url_for('index'))
    
    report_data = stored.get('report_data')
    timestamp = stored.get('timestamp', 'Unknown')
    
    return render_template('results.html', report=report_data, timestamp=timestamp)

@app.route('/batch-results')
def batch_results():
    """Display the results of a batch analysis."""
    batch_id = session.get('batch_id')
    stored = BATCH_RESULTS_STORE.get(batch_id) if batch_id else None
    
    if not stored and BATCH_RESULTS_STORE:
        stored = list(BATCH_RESULTS_STORE.values())[-1]
    
    if not stored:
        flash('No batch analysis results found. Please run a batch analysis first.', 'error')
        return redirect(url_for('batch_analysis'))
    
    results = stored.get('results', [])
    timestamp = stored.get('timestamp', 'Unknown')
    
    return render_template('batch_results.html', results=results, timestamp=timestamp)

@app.route('/data-explorer')
def data_explorer():
    """Display interactive data explorer dashboard for dataset inspection."""
    import pandas as pd
    import json
    
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'training_data.csv')
    if not os.path.exists(data_path):
        from scripts.train_50k_model import load_realistic_dataset
        raw_path = os.path.join(os.path.dirname(__file__), 'data', 'bot_detection_data.csv')
        df = load_realistic_dataset(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
        if not df.empty:
            df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    
    total_records = len(df)
    threat_count = int((df['is_threat'] == 1).sum()) if 'is_threat' in df.columns else 0
    threat_percent = round((threat_count / max(1, total_records)) * 100, 1)
    bot_count = int((df['is_threat'] == 1).sum()) if 'is_threat' in df.columns else 0
    avg_age = int(df['account_age_days'].mean()) if 'account_age_days' in df.columns else 0
    
    # Pass a representative balanced sample (1,500 rows) for smooth high-speed browser rendering
    sample_df = pd.concat([
        df[df['is_threat'] == 1].head(750),
        df[df['is_threat'] == 0].head(750)
    ]).sample(frac=1, random_state=42) if len(df) > 1500 else df
    
    dataset_json = sample_df.to_json(orient='records')
    
    return render_template(
        'data_explorer.html',
        total_records=f"{total_records:,}",
        threat_count=f"{threat_count:,}",
        threat_percent=threat_percent,
        bot_count=f"{bot_count:,}",
        avg_age=f"{avg_age:,}",
        dataset_json=dataset_json
    )

@app.route('/about')
def about():
    """Display information about the project."""
    return render_template('about.html')

@app.route('/model-info')
def model_info():
    """Display model information and live performance metrics from trained model."""
    raw_metrics = getattr(detector.model_trainer, 'metrics', {}) or {}
    
    _FAMILIES = {
        'AdaBoost': 'Boosting Ensemble',
        'Gradient Boosting': 'Boosting Ensemble',
        'HistGradientBoosting': 'Histogram Boosting',
        'Random Forest': 'Bagging Ensemble',
        'Extra Trees': 'Extremely Randomized',
        'Decision Tree': 'Tree Classifier',
        'KNN': 'Instance-Based',
        'Neural Network (MLP)': 'Multi-Layer Perceptron',
        'Logistic Regression': 'Linear Model',
        'Support Vector Machine': 'Kernel SVM',
        'Linear Discriminant': 'Discriminant',
        'Naive Bayes': 'Probabilistic'
    }
    
    leaderboard = []
    for name, m in raw_metrics.items():
        leaderboard.append({
            'name': name,
            'family': _FAMILIES.get(name, 'Machine Learning'),
            'accuracy': m.get('accuracy', 0.0),
            'accuracy_pct': f"{m.get('accuracy', 0.0) * 100:.1f}%",
            'precision': round(m.get('precision', 0.0), 3),
            'recall': round(m.get('recall', 0.0), 3),
            'f1': round(m.get('f1', 0.0), 3),
            'auc': round(m.get('auc', 0.0), 3),
            'is_active': (name == detector.model_name)
        })
    
    # Sort by F1-Score descending
    leaderboard.sort(key=lambda x: x['f1'], reverse=True)
    
    # Extract metadata from trainer if available
    dataset_size = getattr(detector.model_trainer, 'dataset_size', 50000) or 50000
    epochs_count = getattr(detector.model_trainer, 'epochs', 10) or 10
    feature_count = len(getattr(detector.model_trainer, 'feature_names', [])) or 54
    
    return render_template(
        'model_info.html',
        model_name=detector.model_name,
        threat_threshold=detector.threat_threshold,
        dataset_size=dataset_size,
        epochs_count=epochs_count,
        feature_count=feature_count,
        leaderboard=leaderboard
    )

@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
    """Train a new model."""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'training_file' in request.files:
                file = request.files['training_file']
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Train model
                    model_path = detector.train_model(filepath)
                    flash(f'Model trained successfully! Saved to {model_path}', 'success')
                    return redirect(url_for('model_info'))
            
            # Or use default synthetic data
            model_path = detector.train_model()
            flash(f'Model trained on synthetic data! Saved to {model_path}', 'success')
            return redirect(url_for('model_info'))
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}", exc_info=True)
            flash(f'Error training model: {str(e)}', 'error')
            return redirect(url_for('train_model'))
    
    return render_template('train_model.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}", exc_info=True)
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Run the app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)