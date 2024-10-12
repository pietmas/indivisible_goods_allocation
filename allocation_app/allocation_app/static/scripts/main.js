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


document.addEventListener("DOMContentLoaded", function() {
    fetch("/static/data/algorithm_info.json")
        .then(response => response.json()) 
        .then(data => {
            const dropdownContent = document.getElementById('algorithm-dropdown');

            // Loop through the algorithms and dynamically add only the names to the dropdown
            for (const key in data) {
                if (data.hasOwnProperty(key)) {
                    const algorithm = data[key];

                    const algorithmLink = document.createElement('a');
                    const algorithmNameEncoded = encodeURIComponent(algorithm.name); 
                    algorithmLink.href = `/algorithm/${algorithmNameEncoded}/`; 
                    algorithmLink.textContent = algorithm.name; 
                    dropdownContent.appendChild(algorithmLink);
                }
            }
        })
        .catch(error => console.error('Error fetching the JSON:', error));
});