// contact
// Open the consultation pop-up
function openPopup() {
    document.getElementById("popupContainer").style.display = "block";
  }
  
  // Close the consultation pop-up
  function closePopup() {
    document.getElementById("popupContainer").style.display = "none";
  }
  
// query
  // Open the doctor query pop-up
  function openPop() {
    document.getElementById("popupquery").style.display = "block";
  }
  
  // Close the doctor query pop-up
  function closePop() {
    document.getElementById("popupquery").style.display = "none";
  }
  

// bell
  const notificationBtn = document.querySelector('.notification-btn');
const dropdown = document.getElementById('dropdown');

notificationBtn.addEventListener('click', () => {
  const isDropdownVisible = dropdown.style.display === 'block';
  dropdown.style.display = isDropdownVisible ? 'none' : 'block';
});

document.addEventListener('click', (event) => {
  if (!notificationBtn.contains(event.target) && !dropdown.contains(event.target)) {
    dropdown.style.display = 'none';
  }
});
