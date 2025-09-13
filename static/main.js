// Đợi cho toàn bộ tài liệu HTML được tải xong trước khi chạy script
document.addEventListener('DOMContentLoaded', () => {

    // --- LOGIC CHO MENU DI ĐỘNG (HAMBURGER) ---
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');

    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
    }

    // --- LOGIC CHO BỘ CHỌN THEME MÀU ---
    const body = document.body;
    const themePickerToggle = document.getElementById('theme-picker-toggle');
    const themePickerOptions = document.getElementById('theme-picker-options');
    const themeSwatches = document.querySelectorAll('.theme-swatch');

    // 1. Áp dụng theme đã lưu từ lần truy cập trước
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        body.dataset.theme = savedTheme;
    }

    // 2. Xử lý việc bật/tắt menu chọn màu
    if (themePickerToggle && themePickerOptions) {
        themePickerToggle.addEventListener('click', (event) => {
            event.stopPropagation(); // Ngăn sự kiện click lan ra ngoài
            themePickerOptions.classList.toggle('hidden');
        });
    }

    // 3. Xử lý khi người dùng chọn một màu cụ thể
    if (themeSwatches.length > 0) {
        themeSwatches.forEach(swatch => {
            swatch.addEventListener('click', () => {
                const themeName = swatch.dataset.themeName;
                body.dataset.theme = themeName; // Áp dụng theme mới
                localStorage.setItem('theme', themeName); // Lưu lựa chọn vào bộ nhớ
                if (themePickerOptions) {
                    themePickerOptions.classList.add('hidden'); // Ẩn menu đi sau khi chọn
                }
            });
        });
    }

    // 4. Ẩn menu chọn màu nếu người dùng bấm ra ngoài
    document.addEventListener('click', () => {
        if (themePickerOptions && !themePickerOptions.classList.contains('hidden')) {
            themePickerOptions.classList.add('hidden');
        }
    });

});