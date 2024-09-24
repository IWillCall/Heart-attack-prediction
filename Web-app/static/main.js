// main.js
document.addEventListener("DOMContentLoaded", function () {
  const projectButton = document.getElementById("projectDetailsButton");
  const projectDetails = document.getElementById("projectDetails");
  const modelButton = document.getElementById("modelDetailsButton");
  const modelDetails = document.getElementById("modelDetails");

  projectButton.addEventListener("click", function () {
    projectDetails.classList.toggle("open");
  });

  modelButton.addEventListener("click", function () {
    modelDetails.classList.toggle("open");
  });
});
