/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./src/app/**/*.{ts,tsx}",
      "./src/pages/**/*.{ts,tsx}",
      "./src/components/**/*.{ts,tsx}"
    ],
    theme: {
      extend: {},
    },
    plugins: [require("tailwindcss-animate")],
  };
  