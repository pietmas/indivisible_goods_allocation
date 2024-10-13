// allocation_app/static/scripts/main.js

document.addEventListener('DOMContentLoaded', function() {
    const preferencesForm = document.getElementById('preferences-form');
    const loadingDiv = document.getElementById('loading');
    if (preferencesForm && loadingDiv) {
        preferencesForm.addEventListener('submit', function() {
            loadingDiv.style.display = 'block';
        });
    }

    const errorMessages = document.querySelectorAll('.error-message');

    errorMessages.forEach(function(errorSpan) {
        const inputField = errorSpan.previousElementSibling;
        const errorMessage = errorSpan.getAttribute('data-error');

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-error';
        tooltip.innerText = errorMessage;

        errorSpan.parentNode.appendChild(tooltip);

        inputField.addEventListener('focus', function() {
            tooltip.style.display = 'block';
        });

        inputField.addEventListener('blur', function() {
            tooltip.style.display = 'none';
        });
    });
});
