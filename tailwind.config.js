/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/src/**/*.js"
  ],
  theme: {
    extend: {
      colors: {
        'pink': '#f26a8d',
        'green': '#2c666e',
      }
    },
  },
  plugins: [require("daisyui")],
}
