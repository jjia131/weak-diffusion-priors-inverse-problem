window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  var options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
  };

  var carousels = bulmaCarousel.attach('.carousel', options);
  bulmaSlider.attach();

  // Arrow buttons for custom viz-carousel-outer wrappers
  document.querySelectorAll('.viz-carousel-outer').forEach(function (outer) {
    var track = outer.querySelector('.viz-carousel-track');
    var prev  = outer.querySelector('.viz-prev');
    var next  = outer.querySelector('.viz-next');
    if (!track) return;

    function slideWidth() {
      var slide = track.querySelector('.viz-slide');
      return slide ? slide.offsetWidth + 4 : 300;
    }

    if (prev) {
      prev.addEventListener('click', function () {
        track.scrollBy({ left: -slideWidth(), behavior: 'smooth' });
      });
    }
    if (next) {
      next.addEventListener('click', function () {
        track.scrollBy({ left: slideWidth(), behavior: 'smooth' });
      });
    }
  });
});
