// AQ Assessment Questions and Options
const aqQuestions = [
    {
        id: 1,
        text: "You suffer a financial setback. To what extent can you influence this situation?",
        options: [
            "Not at all",
            "Slightly", 
            "Moderately",
            "Mostly",
            "Completely"
        ]
    },
    {
        id: 2,
        text: "You are overlooked for a promotion. To what extent do you feel responsible for improving the situation?",
        options: [
            "Not responsible at all",
            "Slightly responsible",
            "Moderately responsible", 
            "Mostly responsible",
            "Completely responsible"
        ]
    },
    {
        id: 3,
        text: "You are criticized for a big project you just completed. The consequences of this situation will:",
        options: [
            "Affect all aspects of my life",
            "Affect many areas",
            "Affect some areas",
            "Affect only this area", 
            "Be limited to this situation"
        ]
    },
    {
        id: 4,
        text: "You accidentally delete an important email. The consequences of this situation will:",
        options: [
            "Last forever",
            "Last a long time",
            "Last for a while",
            "Be temporary",
            "Quickly pass"
        ]
    },
    {
        id: 5,
        text: "The high-priority project you are working on gets canceled. The consequences of this situation will:",
        options: [
            "Affect all aspects of my life",
            "Affect many areas", 
            "Affect some areas",
            "Affect only this area",
            "Be limited to this situation"
        ]
    },
    {
        id: 6,
        text: "Someone you respect ignores your attempt to discuss an important issue. To what extent do you feel responsible for improving this situation?",
        options: [
            "Not responsible at all",
            "Slightly responsible",
            "Moderately responsible",
            "Mostly responsible",
            "Completely responsible"
        ]
    },
    {
        id: 7,
        text: "People respond unfavorably to your latest ideas. To what extent can you influence this situation?",
        options: [
            "Not at all",
            "Slightly",
            "Moderately", 
            "Mostly",
            "Completely"
        ]
    },
    {
        id: 8,
        text: "You are unable to take a much-needed vacation. The consequences of this situation will:",
        options: [
            "Last forever",
            "Last a long time",
            "Last for a while",
            "Be temporary",
            "Quickly pass"
        ]
    },
    {
        id: 9,
        text: "You hit every red light on your way to an important appointment. The consequences of this situation will:",
        options: [
            "Affect all aspects of my life",
            "Affect many areas",
            "Affect some areas",
            "Affect only this area",
            "Be limited to this situation"
        ]
    },
    {
        id: 10,
        text: "After extensive searching, you cannot find an important document. The consequences of this situation will:",
        options: [
            "Last forever",
            "Last a long time", 
            "Last for a while",
            "Be temporary",
            "Quickly pass"
        ]
    },
    {
        id: 11,
        text: "Your workplace is understaffed. To what extent do you feel responsible for improving this situation?",
        options: [
            "Not responsible at all",
            "Slightly responsible",
            "Moderately responsible",
            "Mostly responsible",
            "Completely responsible"
        ]
    },
    {
        id: 12,
        text: "You miss an important appointment. The consequences of this situation will:",
        options: [
            "Affect all aspects of my life",
            "Affect many areas",
            "Affect some areas",
            "Affect only this area",
            "Be limited to this situation"
        ]
    },
    {
        id: 13,
        text: "Your personal and work obligations are out of balance. To what extent can you influence this situation?",
        options: [
            "Not at all",
            "Slightly",
            "Moderately",
            "Mostly", 
            "Completely"
        ]
    },
    {
        id: 14,
        text: "You never seem to have enough money. The consequences of this situation will:",
        options: [
            "Last forever",
            "Last a long time",
            "Last for a while",
            "Be temporary",
            "Quickly pass"
        ]
    },
    {
        id: 15,
        text: "You are not exercising regularly though you know you should. To what extent can you influence this situation?",
        options: [
            "Not at all",
            "Slightly",
            "Moderately",
            "Mostly",
            "Completely"
        ]
    },
    {
        id: 16,
        text: "Your organization is not meeting its goals. To what extent do you feel responsible for improving this situation?",
        options: [
            "Not responsible at all",
            "Slightly responsible", 
            "Moderately responsible",
            "Mostly responsible",
            "Completely responsible"
        ]
    },
    {
        id: 17,
        text: "Your computer crashed for the third time this week. To what extent can you influence this situation?",
        options: [
            "Not at all",
            "Slightly",
            "Moderately",
            "Mostly",
            "Completely"
        ]
    },
    {
        id: 18,
        text: "The meeting you are in is a total waste of time. To what extent do you feel responsible for improving this situation?",
        options: [
            "Not responsible at all",
            "Slightly responsible",
            "Moderately responsible",
            "Mostly responsible",
            "Completely responsible"
        ]
    },
    {
        id: 19,
        text: "You lost something important to you. The consequences of this situation will:",
        options: [
            "Last forever",
            "Last a long time",
            "Last for a while",
            "Be temporary",
            "Quickly pass"
        ]
    },
    {
        id: 20,
        text: "Your boss adamantly disagrees with your decision. The consequences of this situation will:",
        options: [
            "Affect all aspects of my life",
            "Affect many areas",
            "Affect some areas", 
            "Affect only this area",
            "Be limited to this situation"
        ]
    }
];

// Quiz State Management
class AQQuiz {
    constructor() {
        this.currentQuestion = 0;
        this.answers = [];
        this.totalQuestions = aqQuestions.length;
        
        this.initializeElements();
        this.bindEvents();
    }
    
    initializeElements() {
        // Containers
        this.dashboardContainer = document.getElementById('dashboard-container');
        this.quizContainer = document.getElementById('quiz-container');
        this.resultsContainer = document.getElementById('results-container');
        
        // Dashboard elements
        this.takeQuizBtn = document.getElementById('take-quiz-btn');
        
        // Quiz elements
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.questionNumber = document.getElementById('question-number');
        this.questionText = document.getElementById('question-text');
        this.optionsContainer = document.getElementById('options-container');
        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.finishBtn = document.getElementById('finish-btn');
        
        // Results elements
        this.finalScore = document.getElementById('final-score');
        this.scoreLevel = document.getElementById('score-level');
        this.totalScore = document.getElementById('total-score');
        this.performanceLevel = document.getElementById('performance-level');
        this.completionDate = document.getElementById('completion-date');
        this.scoreDescription = document.getElementById('score-description');
        this.recommendations = document.getElementById('recommendations');
        this.retakeBtn = document.getElementById('retake-btn');
        this.dashboardBtn = document.getElementById('dashboard-btn');
    }
    
    bindEvents() {
        this.takeQuizBtn.addEventListener('click', () => this.startQuiz());
        this.prevBtn.addEventListener('click', () => this.previousQuestion());
        this.nextBtn.addEventListener('click', () => this.nextQuestion());
        this.finishBtn.addEventListener('click', () => this.finishQuiz());
        this.retakeBtn.addEventListener('click', () => this.resetQuiz());
        this.dashboardBtn.addEventListener('click', () => this.showDashboard());
        this.downloadPdfBtn = document.getElementById('download-pdf-btn');
        this.downloadPdfBtn.addEventListener('click', () => this.downloadPDF());
    }
    
    startQuiz() {
        this.currentQuestion = 0;
        this.answers = [];
        this.showQuiz();
        this.displayQuestion();
    }
    
    showDashboard() {
        this.dashboardContainer.style.display = 'block';
        this.quizContainer.classList.remove('active');
        this.resultsContainer.classList.remove('active');
    }
    
    showQuiz() {
        this.dashboardContainer.style.display = 'none';
        this.quizContainer.classList.add('active');
        this.resultsContainer.classList.remove('active');
    }
    
    showResults() {
        this.dashboardContainer.style.display = 'none';
        this.quizContainer.classList.remove('active');
        this.resultsContainer.classList.add('active');
    }
    
    displayQuestion() {
        const question = aqQuestions[this.currentQuestion];
        
        // Update progress
        const progress = ((this.currentQuestion + 1) / this.totalQuestions) * 100;
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = `Question ${this.currentQuestion + 1} of ${this.totalQuestions}`;
        
        // Update question content
        this.questionNumber.textContent = `Question ${this.currentQuestion + 1}`;
        this.questionText.textContent = question.text;
        
        // Generate options
        this.optionsContainer.innerHTML = '';
        question.options.forEach((option, index) => {
            const optionElement = document.createElement('div');
            optionElement.className = 'option';
            optionElement.textContent = option;
            optionElement.dataset.value = index + 1; // 1-5 scoring
            
            // Check if this option was previously selected
            if (this.answers[this.currentQuestion] === index + 1) {
                optionElement.classList.add('selected');
            }
            
            optionElement.addEventListener('click', () => this.selectOption(optionElement, index + 1));
            this.optionsContainer.appendChild(optionElement);
        });
        
        // Update navigation buttons
        this.prevBtn.disabled = this.currentQuestion === 0;
        this.nextBtn.disabled = !this.answers[this.currentQuestion];
        
        // Show/hide finish button
        if (this.currentQuestion === this.totalQuestions - 1) {
            this.nextBtn.style.display = 'none';
            this.finishBtn.style.display = 'inline-flex';
            this.finishBtn.disabled = !this.answers[this.currentQuestion];
        } else {
            this.nextBtn.style.display = 'inline-flex';
            this.finishBtn.style.display = 'none';
        }
    }
    
    selectOption(optionElement, value) {
        // Clear previous selection
        this.optionsContainer.querySelectorAll('.option').forEach(opt => {
            opt.classList.remove('selected');
        });
        
        // Select current option
        optionElement.classList.add('selected');
        this.answers[this.currentQuestion] = value;
        
        // Enable next/finish button
        this.nextBtn.disabled = false;
        this.finishBtn.disabled = false;
    }
    
    previousQuestion() {
        if (this.currentQuestion > 0) {
            this.currentQuestion--;
            this.displayQuestion();
        }
    }
    
    nextQuestion() {
        if (this.currentQuestion < this.totalQuestions - 1) {
            this.currentQuestion++;
            this.displayQuestion();
        }
    }
    
    finishQuiz() {
        // Check if all questions are answered
        if (this.answers.length < this.totalQuestions || this.answers.includes(undefined)) {
            alert('Please answer all questions before finishing the quiz.');
            return;
        }
        
        this.calculateResults();
        this.showResults();
    }
    
    calculateResults() {
        // Calculate total score
        const totalScore = this.answers.reduce((sum, answer) => sum + answer, 0);
        const aqScore = totalScore * 2; // ARP = Total × 2
        
        // Determine level
        let level, description, recommendations;
        
        if (aqScore >= 180) {
            level = 'Exceptional';
            description = 'You demonstrate exceptional adversity quotient. You have outstanding ability to navigate through challenges, maintain resilience, and turn obstacles into opportunities. Your response to adversity is highly effective and adaptive.';
            recommendations = [
                'Continue leveraging your strong adversity skills in leadership roles',
                'Mentor others who may struggle with challenges',
                'Share your resilience strategies with your team',
                'Take on complex projects that others might find overwhelming',
                'Consider roles that involve crisis management or change leadership'
            ];
        } else if (aqScore >= 160) {
            level = 'High';
            description = 'You have a high adversity quotient, showing strong resilience and effective coping strategies. You handle most challenges well and recover quickly from setbacks. Your ability to maintain perspective during difficult times is commendable.';
            recommendations = [
                'Continue building on your strong foundation of resilience',
                'Practice mindfulness techniques to enhance your coping strategies',
                'Seek challenging assignments to further develop your skills',
                'Consider mentoring others in resilience building',
                'Focus on maintaining work-life balance during stressful periods'
            ];
        } else if (aqScore >= 140) {
            level = 'Above Average';
            description = 'Your adversity quotient is above average. You generally handle challenges well, though there may be some areas where you could strengthen your response to adversity. You show good resilience in most situations.';
            recommendations = [
                'Identify specific areas where you feel less confident facing challenges',
                'Practice reframing negative situations into learning opportunities',
                'Develop a stronger support network for difficult times',
                'Work on building patience and persistence in challenging situations',
                'Consider stress management techniques like meditation or exercise'
            ];
        } else if (aqScore >= 120) {
            level = 'Average';
            description = 'You have an average adversity quotient. While you can handle routine challenges, you may struggle with unexpected or significant adversities. There is good potential for improvement in your resilience and coping strategies.';
            recommendations = [
                'Focus on developing a growth mindset toward challenges',
                'Practice breaking down large problems into smaller, manageable parts',
                'Build stronger problem-solving skills through practice',
                'Seek feedback from others on how they handle similar challenges',
                'Consider reading books or taking courses on resilience building'
            ];
        } else if (aqScore >= 100) {
            level = 'Below Average';
            description = 'Your adversity quotient is below average. You may find challenges overwhelming and struggle to maintain perspective during difficult times. With focused effort and the right strategies, you can significantly improve your resilience.';
            recommendations = [
                'Start with small challenges to build confidence gradually',
                'Develop a daily routine that includes stress-reduction activities',
                'Practice positive self-talk and challenge negative thinking patterns',
                'Seek support from friends, family, or a counselor when facing difficulties',
                'Focus on building one coping skill at a time'
            ];
        } else {
            level = 'Needs Improvement';
            description = 'Your adversity quotient indicates significant room for improvement. Challenges may feel overwhelming, and you might struggle to see solutions or maintain hope during difficult times. This is an opportunity to develop stronger resilience skills.';
            recommendations = [
                'Consider working with a counselor or coach to develop coping strategies',
                'Start with very small challenges to build success experiences',
                'Practice daily gratitude and positive thinking exercises',
                'Build a strong support network of understanding friends and family',
                'Focus on physical wellness as a foundation for mental resilience'
            ];
        }
        
        // Update results display
        this.finalScore.textContent = aqScore;
        this.scoreLevel.textContent = level;
        this.totalScore.textContent = `${totalScore}/100`;
        this.performanceLevel.textContent = level;
        this.completionDate.textContent = new Date().toLocaleDateString();
        this.scoreDescription.textContent = description;
        
        // Update recommendations
        this.recommendations.innerHTML = '<ul>' + 
            recommendations.map(rec => `<li>${rec}</li>`).join('') + 
            '</ul>';
        
        // Save results to backend
        this.saveResultsToBackend({
            score: aqScore,
            totalScore: totalScore,
            level: level,
            answers: this.answers,
            completionDate: new Date().toISOString()
        });
        
        // Store results in localStorage for potential future use
        this.saveResultsLocally({
            score: aqScore,
            totalScore: totalScore,
            level: level,
            answers: this.answers,
            completionDate: new Date().toISOString()
        });
    }
    
    downloadPDF() {
    window.open('/student/download-pdf', '_blank');
}

    saveResultsToBackend(results) {
        // Implement backend saving here
        // This is a placeholder - replace with your actual backend endpoint
        fetch('/api/save-aq-results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(results)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Results saved successfully:', data);
        })
        .catch(error => {
            console.error('Error saving results:', error);
            // Handle error gracefully - maybe show a message to user
        });
    }
    
    saveResultsLocally(results) {
        try {
            localStorage.setItem('aq_results', JSON.stringify(results));
            console.log('Results saved locally');
        } catch (error) {
            console.error('Error saving results locally:', error);
        }
    }
    
    resetQuiz() {
        this.currentQuestion = 0;
        this.answers = [];
        this.startQuiz();
    }
}

// Initialize the quiz when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const quiz = new AQQuiz();
    
    // Check if there are previous results to display
    try {
        const savedResults = localStorage.getItem('aq_results');
        if (savedResults) {
            const results = JSON.parse(savedResults);
            // You could add logic here to show previous results or indicate if user has taken the quiz before
            console.log('Previous results found:', results);
        }
    } catch (e) {
        // No previous results or localStorage not available
        console.log('No previous results found');
    }
});