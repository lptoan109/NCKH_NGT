// static/main.js

document.addEventListener('DOMContentLoaded', () => {
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');

    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            // SỬA DÒNG NÀY: đổi 'is-active' thành 'active'
            navMenu.classList.toggle('active');
        });
    }
});