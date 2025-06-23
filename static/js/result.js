// Debug Result Page JavaScript - result.js

document.addEventListener('DOMContentLoaded', function() {
    console.log('=== RESULT PAGE DEBUG ===');
    
    // Check if we're on the right page
    console.log('Current URL:', window.location.href);
    console.log('Page title:', document.title);
    
    // Check for results data
    const table = document.querySelector('table');
    const resultsSection = document.querySelector('.results-section');
    const noResultsMessage = document.querySelector('.no-results') || document.querySelector('[class*="no-result"]');
    
    console.log('Table found:', !!table);
    console.log('Results section found:', !!resultsSection);
    console.log('No results message found:', !!noResultsMessage);
    
    if (table) {
        const rows = table.querySelectorAll('tbody tr');
        console.log('Table rows found:', rows.length);
        if (rows.length > 0) {
            console.log('Sample row content:', rows[0].innerHTML);
        }
    }
    
    // Check for template variables (if passed from Flask)
    if (typeof results !== 'undefined') {
        console.log('Results variable found:', results);
        console.log('Results count:', results.length);
    } else {
        console.log('No results variable found in JavaScript');
    }
    
    if (typeof filename !== 'undefined') {
        console.log('Filename:', filename);
    }
    
    // Get button elements
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const exportCsvBtn = document.getElementById('exportCsvBtn');
    const backToDashboardBtn = document.getElementById('backToDashboardBtn');
    
    console.log('Buttons found:', {
        downloadPdf: !!downloadPdfBtn,
        exportCsv: !!exportCsvBtn,
        backToDashboard: !!backToDashboardBtn
    });

    // Download PDF functionality with debug
    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener('click', function() {
            console.log('PDF download clicked');
            
            // Add loading state
            const originalText = this.innerHTML;
            this.innerHTML = '<span class="loading"></span> Generating PDF...';
            this.disabled = true;

            // Check if route exists before redirecting
            fetch('/download-pdf', { method: 'HEAD' })
                .then(response => {
                    console.log('PDF route response:', response.status);
                    if (response.ok || response.status === 405) { // 405 = Method not allowed (GET vs HEAD)
                        window.location.href = '/download-pdf';
                    } else {
                        console.error('PDF route not found');
                        alert('PDF download route not available');
                    }
                })
                .catch(error => {
                    console.error('Error checking PDF route:', error);
                    // Try anyway
                    window.location.href = '/download-pdf';
                });

            // Reset button state after a delay
            setTimeout(() => {
                this.innerHTML = originalText;
                this.disabled = false;
            }, 3000);
        });
    }

    // Export CSV functionality with debug
    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', function() {
            console.log('CSV export clicked');
            
            // Add loading state
            const originalText = this.innerHTML;
            this.innerHTML = '<span class="loading"></span> Exporting...';
            this.disabled = true;

            // Check if route exists
            fetch('/export-csv', { method: 'HEAD' })
                .then(response => {
                    console.log('CSV route response:', response.status);
                    if (response.ok || response.status === 405) {
                        window.location.href = '/export-csv';
                    } else {
                        console.error('CSV route not found');
                        alert('CSV export route not available');
                    }
                })
                .catch(error => {
                    console.error('Error checking CSV route:', error);
                    window.location.href = '/export-csv';
                });

            // Reset button state after a delay
            setTimeout(() => {
                this.innerHTML = originalText;
                this.disabled = false;
            }, 2000);
        });
    }

    // Back to Dashboard functionality with multiple route attempts
    if (backToDashboardBtn) {
        backToDashboardBtn.addEventListener('click', function() {
            console.log('Back to dashboard clicked');
            
            // Try different possible dashboard routes
            const possibleRoutes = [
                '/teacher/dashboard',
                '/dashboard', 
                '/teacher-dashboard',
                '/teacher',
                '/'
            ];
            
            // Try each route until one works
            let routeIndex = 0;
            
            function tryNextRoute() {
                if (routeIndex >= possibleRoutes.length) {
                    console.error('No valid dashboard route found');
                    alert('Dashboard route not found');
                    return;
                }
                
                const route = possibleRoutes[routeIndex];
                console.log(`Trying route: ${route}`);
                
                fetch(route, { method: 'HEAD' })
                    .then(response => {
                        if (response.ok) {
                            console.log(`Found working route: ${route}`);
                            window.location.href = route;
                        } else {
                            routeIndex++;
                            tryNextRoute();
                        }
                    })
                    .catch(error => {
                        console.log(`Route ${route} failed:`, error);
                        routeIndex++;
                        tryNextRoute();
                    });
            }
            
            tryNextRoute();
        });
    }

    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && backToDashboardBtn) {
            console.log('Escape key pressed');
            backToDashboardBtn.click();
        }
        
        if (e.ctrlKey && e.key === 's' && exportCsvBtn) {
            e.preventDefault();
            console.log('Ctrl+S pressed');
            exportCsvBtn.click();
        }
        
        if (e.ctrlKey && e.key === 'p' && downloadPdfBtn) {
            e.preventDefault();
            console.log('Ctrl+P pressed');
            downloadPdfBtn.click();
        }
    });

    console.log('=== DEBUG INITIALIZATION COMPLETE ===');
});