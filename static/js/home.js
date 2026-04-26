// Home page animations and interactivity

document.addEventListener("DOMContentLoaded", () => {
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: "0rem 0rem -3.125rem 0rem"
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
