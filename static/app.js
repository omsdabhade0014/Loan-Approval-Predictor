/* Loan Approval Predictor - front-end behaviour
 *  1. Shows a short, step-by-step loading sequence after "Run assessment" is clicked,
 *     then submits the form (the server work itself is fast; the pause is for the user).
 *  2. Asks for confirmation before deleting a history record.
 */
(function () {
  var STEP_MS = 600; // time each loading step is shown

  var form = document.getElementById("predict-form");
  var loader = document.getElementById("loader");
  var fill = document.getElementById("loader-fill");
  var steps = loader ? loader.querySelectorAll("[data-step]") : [];

  function runLoader(onDone) {
    Array.prototype.forEach.call(steps, function (s) {
      s.classList.remove("is-current", "is-done");
    });
    loader.classList.add("is-open");

    // Reset and restart the progress bar
    fill.style.transition = "none";
    fill.style.width = "0%";
    void fill.offsetWidth; // force a reflow so the reset takes effect

    var total = steps.length * STEP_MS + 400;
    fill.style.transition = "width " + total + "ms linear";
    fill.style.width = "100%";

    Array.prototype.forEach.call(steps, function (step, i) {
      setTimeout(function () {
        Array.prototype.forEach.call(steps, function (other, j) {
          if (j < i) {
            other.classList.remove("is-current");
            other.classList.add("is-done");
          }
        });
        step.classList.add("is-current");
      }, i * STEP_MS);
    });

    setTimeout(onDone, total);
  }

  if (form && loader) {
    form.addEventListener("submit", function (event) {
      // With native validation on, an invalid form never reaches this handler;
      // the check is a safety net.
      if (!form.checkValidity()) return;
      event.preventDefault();
      runLoader(function () {
        form.submit(); // submit() does not fire the submit event again, so no loop
      });
    });
  }

  // If the user navigates Back to the form, hide the overlay that was left open
  window.addEventListener("pageshow", function () {
    if (loader) loader.classList.remove("is-open");
  });

  // Confirm before deleting a history record
  Array.prototype.forEach.call(document.querySelectorAll("form[data-confirm]"), function (f) {
    f.addEventListener("submit", function (event) {
      if (!window.confirm(f.getAttribute("data-confirm"))) event.preventDefault();
    });
  });
})();
