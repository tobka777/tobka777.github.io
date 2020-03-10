const g = document.querySelector("g");
const circle = document.querySelector("circle");
const HAPPY = "Happy";
const HAPPY_PATH = "M 150 200 Q 225 300 300 200";
const SAD = "Sad";
const SAD_PATH = "M 150 200 Q 225 100 300 200";

function change() {
  var text = document.getElementById("label_sad_happy").textContent;
  var path = document.getElementById("smiley_path").getAttribute("d");
  document.getElementById("label_sad_happy").textContent =
    text == HAPPY ? SAD : HAPPY;
  document
    .getElementById("smiley_path")
    .setAttribute("d", path == HAPPY_PATH ? SAD_PATH : HAPPY_PATH);
}
