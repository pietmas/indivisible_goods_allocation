// allocation_app/static/scripts/main.js

document.addEventListener('DOMContentLoaded', function() {
    const preferencesForm = document.getElementById('preferences-form');
    const loadingDiv = document.getElementById('loading');
    if (preferencesForm && loadingDiv) {
        preferencesForm.addEventListener('submit', function() {
            // Show loading animation
            loadingDiv.style.display = 'block';
        });
    }

    // Error Tooltip Functionality
    const errorMessages = document.querySelectorAll('.error-message');

    errorMessages.forEach(function(errorSpan) {
        const inputField = errorSpan.previousElementSibling;
        const errorMessage = errorSpan.getAttribute('data-error');

        // Create a tooltip element
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-error';
        tooltip.innerText = errorMessage;

        // Insert the tooltip into the DOM
        errorSpan.parentNode.appendChild(tooltip);

        // Show tooltip on focus
        inputField.addEventListener('focus', function() {
            tooltip.style.display = 'block';
        });

        // Hide tooltip on blur
        inputField.addEventListener('blur', function() {
            tooltip.style.display = 'none';
        });
    });
});
