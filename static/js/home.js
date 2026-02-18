// Home page animations and interactivity

document.addEventListener("DOMContentLoaded", () => {
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px"
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe feature cards
    document.querySelectorAll(".feature-card").forEach((card) => {
        observer.observe(card);
    });

    // Observe stat boxes
    document.querySelectorAll(".stat-box").forEach((box) => {
        observer.observe(box);
    });
});

// Add visible class styling
const style = document.createElement("style");
style.textContent = `
    .feature-card,
    .stat-box {
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.6s ease, transform 0.6s ease;
    }

    .feature-card.visible,
    .stat-box.visible {
        opacity: 1;
        transform: translateY(0);
    }
`;
document.head.appendChild(style);
