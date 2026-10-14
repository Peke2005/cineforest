// URL pública de la API desplegada en Render (sin barra final).
// Para probar en local: abre index.html?api=http://localhost:8000
window.API_URL = new URLSearchParams(location.search).get("api") || "https://cineforest-api.onrender.com";
