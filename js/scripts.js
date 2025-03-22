document.addEventListener("DOMContentLoaded", function () {
  // If you have collapsible menu items, they can stay here; otherwise, remove them
  const collapsibleLinks = document.querySelectorAll(".menu .collapsible");
  collapsibleLinks.forEach(link => {
    link.addEventListener("click", function (event) {
      event.preventDefault();
      link.classList.toggle("active");
      const submenu = link.nextElementSibling;
      if (submenu) {
        submenu.style.display =
          submenu.style.display === "block" ? "none" : "block";
      }
    });
  });

  // Allow "Enter" key to send messages
  document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent accidental form submission
      sendMessage();
    }
  });

  // Attach click event to the send button
  document.getElementById("send-button").addEventListener("click", sendMessage);
});

// Function to handle sending messages
async function sendMessage() {
  const userInputField = document.getElementById("user-input");
  const userInput = userInputField.value.trim();

  if (!userInput) {
    alert("Please enter a question.");
    return;
  }

  // Clear the input after sending
  userInputField.value = "";

  // Display the user's message in the chat
  addMessageToChat("You", userInput);

  try {
    // Send a POST request to your local Flask app's /ask endpoint
    const response = await fetch("http://127.0.0.1:5000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      credentials: "include",  // Ensure session cookies are sent
      body: JSON.stringify({ question: userInput })
    });
    
    
    if (response.ok) {
      const data = await response.json();
      addMessageToChat("Bot", data.answer);
    } else {
      addMessageToChat("Bot", "Sorry, something went wrong. Please try again.");
    }
  } catch (error) {
    console.error("Error:", error);
    addMessageToChat("Bot", "There was an error connecting to the server.");
  }
}

// Function to add messages to the chat box with proper formatting
function addMessageToChat(sender, message) {
  const chatMessages = document.getElementById("chat-messages");
  const messageElement = document.createElement("div");
  messageElement.classList.add("message");

  // Replace newline characters with <br> tags for proper display.
  const formattedMessage = message.replace(/\n/g, "<br>");

  messageElement.innerHTML = `<strong>${sender}:</strong><br>${formattedMessage}`;
  chatMessages.appendChild(messageElement);

  // Scroll to the latest message
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
