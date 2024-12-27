// Toggle between dark and light modes
function toggleMode() {
    var body = document.body;
    body.classList.toggle("dark-mode");

    var lightModeIcon = document.getElementById("light-mode-icon");
    var darkModeIcon = document.getElementById("dark-mode-icon");

    if (body.classList.contains("dark-mode")) {
        lightModeIcon.style.display = "none";
        darkModeIcon.style.display = "block";
    } else {
        lightModeIcon.style.display = "block";
        darkModeIcon.style.display = "none";
    }
}

document.addEventListener("DOMContentLoaded", function () {
const sections = document.querySelectorAll('.section');

function checkVisibility() {
    const windowHeight = window.innerHeight;
    sections.forEach(section => {
        const sectionTop = section.getBoundingClientRect().top;
        if (sectionTop < windowHeight * 0.8) { // 80% of the viewport height
            section.classList.add('visible');
            section.classList.remove('hidden');
        }
    });
}

// Initially, add the 'hidden' class to all sections
sections.forEach(section => {
    section.classList.add('hidden');
});

// Check visibility on scroll and resize events
window.addEventListener('scroll', checkVisibility);
window.addEventListener('resize', checkVisibility);

// Call checkVisibility on load to handle sections already in view
checkVisibility();
});