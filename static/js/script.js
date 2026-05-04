// Navigation
let menuIcon = document.querySelector("#menu-icon");
let navbar = document.querySelector(".navbar");
const isHomePage = window.location.pathname === "/";

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
    if (isHomePage) {
        updateActiveNavLink();
    }
});

function updateActiveNavLink() {
    if (!isHomePage) {
        return;
    }

    let sections = document.querySelectorAll("section");
    let navLinks = document.querySelectorAll(".navbar .nav-link");
    let activeSectionId = null;
    let top = window.scrollY;

    sections.forEach((sec) => {
        let offset = sec.offsetTop - 200;
        let height = sec.offsetHeight;
        let id = sec.getAttribute("id");

        if (id && top >= offset && top < offset + height) {
            activeSectionId = id;
        }
    });

    navLinks.forEach((link) => {
        link.classList.remove("active");
    });

    if (activeSectionId) {
        document.querySelector(`.navbar .nav-link[href="/#${activeSectionId}"]`)?.classList.add("active");
    } else {
        document.querySelector('.navbar .nav-link[href="/"]')?.classList.add("active");
    }
}

function setActiveNavForCurrentPath() {
    const navLinks = document.querySelectorAll(".navbar .nav-link");
    const path = window.location.pathname;

    navLinks.forEach((link) => {
        link.classList.remove("active");
    });

    if (path.startsWith("/dashboard")) {
        document.querySelector('.navbar .nav-link[href="/dashboard"]')?.classList.add("active");
        return;
    }

    if (path.startsWith("/history")) {
        document.querySelector('.navbar .nav-link[href="/history"]')?.classList.add("active");
        return;
    }

    if (path.startsWith("/analytics")) {
        document.querySelector('.navbar .nav-link[href="/analytics"]')?.classList.add("active");
        return;
    }

    if (isHomePage) {
        document.querySelector('.navbar .nav-link[href="/"]')?.classList.add("active");
    }
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
    setActiveNavForCurrentPath();
    if (isHomePage) {
        updateActiveNavLink();
    }

    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    // Handle URL parameters for dashboard tab navigation
    const urlParams = new URLSearchParams(window.location.search);
    const tabParam = urlParams.get('tab');
    
    if (tabParam && document.getElementById(tabParam)) {
        // Switch to the tab specified in URL
        setTimeout(() => {
            const targetBtn = document.querySelector(`[data-tab="${tabParam}"]`);
            if (targetBtn) {
                // Hide all tabs
                tabContents.forEach((tab) => tab.classList.remove("active"));
                // Remove active class from all buttons
                tabBtns.forEach((b) => b.classList.remove("active"));
                // Show selected tab
                document.getElementById(tabParam)?.classList.add("active");
                targetBtn.classList.add("active");
                
                // Scroll to dashboard section
                document.querySelector('.dashboard-container')?.scrollIntoView({ behavior: 'smooth' });
            }
        }, 100);
    }

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
            
            // Update URL without page reload
            window.history.pushState({ tab: tabId }, '', `/dashboard?tab=${tabId}`);
        });
    });
});
