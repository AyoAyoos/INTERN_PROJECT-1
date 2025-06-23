// loading.js - Loading UI functionality (Fixed - No Browser Warnings)

class LoadingUI {
    constructor() {
        this.overlay = null;
        this.progressInterval = null;
        this.isUploading = false;
        this.init();
    }

    init() {
        // Create loading overlay HTML
        this.createLoadingOverlay();
        // Bind form submission
        this.bindFormSubmission();
    }

    createLoadingOverlay() {
        // Create overlay element
        this.overlay = document.createElement('div');
        this.overlay.className = 'loading-overlay';
        this.overlay.id = 'loadingOverlay';
        
        // Create loading content
        this.overlay.innerHTML = `
            <div class="loading-content">
                <div class="spinner"></div>
                <div class="loading-text">Processing Your File</div>
                <div class="loading-subtext" id="loadingSubtext">Initializing BERT model</div>
                <div class="progress-container">
                    <div class="progress-bar" id="progressBar"></div>
                </div>
                <div class="loading-steps">
                    <div class="step" id="step1">
                        <span class="status-indicator"></span>
                        File uploaded
                    </div>
                    <div class="step" id="step2">
                        <span class="status-indicator"></span>
                        Loading AI model
                    </div>
                    <div class="step" id="step3">
                        <span class="status-indicator"></span>
                        Analyzing questions
                    </div>
                    <div class="step" id="step4">
                        <span class="status-indicator"></span>
                        Generating results
                    </div>
                </div>
            </div>
        `;
        
        // Add to body
        document.body.appendChild(this.overlay);
    }

    bindFormSubmission() {
        const form = document.querySelector('form[action="/upload"]');
        if (form) {
            form.addEventListener('submit', (e) => {
                // Validate file selection
                const fileInput = form.querySelector('input[type="file"]');
                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('Please select a file first!');
                    e.preventDefault();
                    return;
                }

                // Validate file type
                const file = fileInput.files[0];
                const allowedTypes = ['.csv', '.xlsx', '.xls'];
                const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
                
                if (!allowedTypes.includes(fileExtension)) {
                    alert('Please select a valid CSV or Excel file!');
                    e.preventDefault();
                    return;
                }

                // Set uploading state
                this.isUploading = true;
                
                // Show loading UI immediately
                this.showLoading();
                
                // Start progress simulation
                this.startProgressSimulation();
                
                console.log('File upload started:', file.name);
                
                // Let the form submit normally - don't prevent it
                // The loading modal will show while the upload happens
            });
        }
    }

    showLoading() {
        if (this.overlay) {
            this.overlay.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    }

    hideLoading() {
        if (this.overlay) {
            this.overlay.style.display = 'none';
            document.body.style.overflow = '';
        }
        this.isUploading = false;
        this.stopProgressSimulation();
    }

    startProgressSimulation() {
        let step = 1;
        let progress = 0;
        const subtexts = [
            "Uploading file",
            "Initializing BERT model",
            "Processing questions", 
            "Analyzing with AI",
            "Calculating confidence scores",
            "Finalizing results"
        ];

        // Activate first step
        this.activateStep(1);
        
        this.progressInterval = setInterval(() => {
            progress += Math.random() * 12 + 3;
            
            if (progress > 100) progress = 100;
            
            // Update progress bar
            const progressBar = document.getElementById('progressBar');
            if (progressBar) {
                progressBar.style.width = progress + '%';
            }
            
            // Update subtext
            const subtextElement = document.getElementById('loadingSubtext');
            if (subtextElement && subtexts[Math.floor(progress / 17)]) {
                subtextElement.textContent = subtexts[Math.floor(progress / 17)];
            }
            
            // Activate steps based on progress
            if (progress > 25 && step < 2) {
                this.activateStep(2);
                step = 2;
            } else if (progress > 50 && step < 3) {
                this.activateStep(3);
                step = 3;
            } else if (progress > 75 && step < 4) {
                this.activateStep(4);
                step = 4;
            }
            
            // Stop at 95% to wait for actual completion
            if (progress >= 95) {
                this.stopProgressSimulation();
            }
            
        }, 400);
    }

    stopProgressSimulation() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    activateStep(stepNumber) {
        const step = document.getElementById(`step${stepNumber}`);
        if (step) {
            step.classList.add('active');
            // Mark previous steps as completed
            for (let i = 1; i < stepNumber; i++) {
                const prevStep = document.getElementById(`step${i}`);
                if (prevStep) {
                    prevStep.classList.add('completed');
                    prevStep.classList.remove('active');
                }
            }
        }
    }

    // Method to be called when classification is complete
    onComplete() {
        // Complete the progress bar
        const progressBar = document.getElementById('progressBar');
        const subtextElement = document.getElementById('loadingSubtext');
        
        if (progressBar) progressBar.style.width = '100%';
        if (subtextElement) subtextElement.textContent = 'Complete! Redirecting';
        
        // Activate all steps
        for (let i = 1; i <= 4; i++) {
            this.activateStep(i);
        }
        
        // Hide loading after a short delay
        setTimeout(() => {
            this.hideLoading();
        }, 1000);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.loadingUI = new LoadingUI();
    
    // Handle page visibility change (user switches tabs)
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            console.log('User switched tabs during upload');
        } else {
            console.log('User returned to tab');
        }
    });
    
    // REMOVED: beforeunload event listener that was causing the "Leave site?" dialog
    // The loading modal will show immediately when upload starts
    // No browser warnings will appear
});