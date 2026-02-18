// Navigation
let menuIcon = document.querySelector("#menu-icon");
let navbar = document.querySelector(".navbar");

if (menuIcon) {
    menuIcon.onclick = () => {
        menuIcon.classList.toggle("fa-xmark");
        navbar.classList.toggle("active");
    };
}

// Sticky Header
window.addEventListener("scroll", () => {
    let header = document.querySelector(".header");
    header.classList.toggle("sticky", window.scrollY > 100);

    // Close menu on scroll
    if (menuIcon && navbar) {
        menuIcon.classList.remove("fa-xmark");
        navbar.classList.remove("active");
    }

    // Update active nav link
    updateActiveNavLink();
});

function updateActiveNavLink() {
    let sections = document.querySelectorAll("section");
    let navLinks = document.querySelectorAll(".navbar .nav-link");

    sections.forEach((sec) => {
        let top = window.scrollY;
        let offset = sec.offsetTop - 200;
        let height = sec.offsetHeight;
        let id = sec.getAttribute("id");

        if (top >= offset && top < offset + height) {
            navLinks.forEach((link) => {
                link.classList.remove("active");
            });
            document.querySelector(`.navbar .nav-link[href="#${id}"]`)?.classList.add("active");
        }
    });
}

// Scroll to top
let scrollTopBtn = document.getElementById("scroll-top");
if (scrollTopBtn) {
    scrollTopBtn.addEventListener("click", (e) => {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
}

// Close mobile menu when clicking on a link
document.querySelectorAll(".navbar .nav-link").forEach((link) => {
    link.addEventListener("click", () => {
        menuIcon?.classList.remove("fa-xmark");
        navbar?.classList.remove("active");
    });
});

// Dashboard page - tab switching logic
document.addEventListener("DOMContentLoaded", () => {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const tabId = btn.dataset.tab;

            // Hide all tabs
            tabContents.forEach((tab) => tab.classList.remove("active"));

            // Remove active class from all buttons
            tabBtns.forEach((b) => b.classList.remove("active"));

            // Show selected tab
            document.getElementById(tabId)?.classList.add("active");
            btn.classList.add("active");
        });
    });
});
