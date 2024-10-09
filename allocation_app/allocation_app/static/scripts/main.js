
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

document.addEventListener('DOMContentLoaded', function() {
    // Get all error messages
    const errorMessages = document.querySelectorAll('.error-message');

    errorMessages.forEach(function(errorSpan) {
        // Get the corresponding input field
        const inputField = errorSpan.previousElementSibling;

        // Get the error message from the data attribute
        const errorMessage = errorSpan.getAttribute('data-error');

        // Create a tooltip element
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-error';
        tooltip.innerText = errorMessage;
        tooltip.style.display = 'none'; // Hide by default

        // Insert the tooltip into the DOM
        inputField.parentNode.insertBefore(tooltip, inputField.nextSibling);

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