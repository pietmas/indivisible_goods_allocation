
document.addEventListener('DOMContentLoaded', function() {
    const preferencesForm = document.getElementById('preferences-form');
    const loadingDiv = document.getElementById('loading');
    if (preferencesForm && loadingDiv) {
        preferencesForm.addEventListener('submit', function() {
            // Show loading animation
            loadingDiv.style.display = 'block';
        });
    }
});