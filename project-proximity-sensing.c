#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>


// OLED display size
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);


// Ultrasonic pins (1 common trigger + 4 echo pins)
#define SHARED_TRIG D5
#define F_ECHO D6
#define B_ECHO D7
#define L_ECHO D0
#define R_ECHO D8


// Buzzer pin
#define BUZZER D4


// LED indications
#define LED_WARNING D3    // Warning (Green)
#define LED_DANGER  3     // Danger (Yellow - RX pin)
#define LED_CRITICAL 1    // Critical (Red - TX pin)


// ---------------------------------------------------------
// Function to read distance from an ultrasonic sensor
// Uses common trigger and measures echo duration
// Returns distance in centimeters
long readSensor(int echoPin) {
  digitalWrite(SHARED_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(SHARED_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(SHARED_TRIG, LOW);


  long duration = pulseIn(echoPin, HIGH, 25000); // max 25ms timeout
  long distance = duration * 0.034 / 2;          // convert to cm


  if (distance == 0) return 999; // if no reading, return large value
  return distance;
}


// ---------------------------------------------------------
// Function to update OLED display with readings
// Shows Front/Back/Left/Right distances
// Adds "!!" indicators for warning zones
void updateDisplay(int front, int back, int left, int right) {
  display.clearDisplay();


  // Header box
  display.fillRect(0, 0, 128, 14, WHITE);
  display.setTextSize(1);
  display.setTextColor(BLACK);
  display.setCursor(15, 3);
  display.println("Distance (cm)");


  display.setTextColor(WHITE);


  // FRONT
  display.setCursor(5, 20);
  display.print("FRONT: ");
  display.print(front);
  if (front < 5) display.print(" !!!");
  else if (front < 8) display.print(" !!");


  // BACK
  display.setCursor(5, 32);
  display.print("BACK : ");
  display.print(back);
  if (back < 5) display.print(" !!!");
  else if (back < 8) display.print(" !!");


  // LEFT
  display.setCursor(5, 44);
  display.print("LEFT : ");
  display.print(left);
  if (left < 5) display.print(" !!!");
  else if (left < 8) display.print(" !!");


  // RIGHT
  display.setCursor(5, 56);
  display.print("RIGHT: ");
  display.print(right);
  if (right < 5) display.print(" !!!");
  else if (right < 8) display.print(" !!");


  display.display();
}


// ---------------------------------------------------------
// Setup function runs once - initializes pins, LEDs, OLED
void setup() {
  pinMode(SHARED_TRIG, OUTPUT);
  pinMode(F_ECHO, INPUT);
  pinMode(B_ECHO, INPUT);
  pinMode(L_ECHO, INPUT);
  pinMode(R_ECHO, INPUT);


  pinMode(BUZZER, OUTPUT);


  pinMode(LED_WARNING, OUTPUT);
  pinMode(LED_DANGER, OUTPUT);
  pinMode(LED_CRITICAL, OUTPUT);


  // Turn off all LEDs initially
  digitalWrite(LED_WARNING, LOW);
  digitalWrite(LED_DANGER, LOW);
  digitalWrite(LED_CRITICAL, LOW);


  // Initialize OLED
  Wire.begin(D2, D1);                     
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);


  // Startup welcome screen
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(25, 20);
  display.println("VEHICLE ALERT");
  display.setCursor(46, 40);
  display.println("SYSTEM");
  display.display();
  delay(1500);
}


// ---------------------------------------------------------
// Loop runs continuously - reads sensors and gives alerts
void loop() {


  // Read all 4 sensors with small delay in between
  int front = readSensor(F_ECHO);
  delay(20);
  int back = readSensor(B_ECHO);
  delay(20);
  int left = readSensor(L_ECHO);
  delay(20);
  int right = readSensor(R_ECHO);
  delay(20);


  // Determine the nearest direction
  int minDist = front;
  String dir = "FRONT";


  if (back < minDist) { minDist = back; dir = "BACK"; }
  if (left < minDist) { minDist = left; dir = "LEFT"; }
  if (right < minDist) { minDist = right; dir = "RIGHT"; }


  // Reset all alerts
  digitalWrite(LED_WARNING, LOW);
  digitalWrite(LED_DANGER, LOW);
  digitalWrite(LED_CRITICAL, LOW);
  noTone(BUZZER);


  // ---------------------- CRITICAL ALERT (<=2 cm) ----------------------
  if (minDist <= 2) {
    digitalWrite(LED_CRITICAL, HIGH);
    tone(BUZZER, 700);  // continuous high-priority beep


    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(25, 20);
    display.println("!!! ALERT !!!");


    display.setCursor(2, 40);
    display.print("!! FROM ");
    display.print(dir);
    display.println(" SIDE !!");


    display.display();
    return;
  }


  // ---------------------- DANGER ALERT (<5 cm) ----------------------
  bool danger = (front < 5 || back < 5 || left < 5 || right < 5);


  if (danger) {
    updateDisplay(front, back, left, right);


    digitalWrite(LED_DANGER, HIGH);
    tone(BUZZER, 1500); // fast beep
    delay(300);


    digitalWrite(LED_DANGER, LOW);
    noTone(BUZZER);
    delay(300);
    return;
  }
  // ---------------------- WARNING ALERT (<8 cm) ----------------------
  bool warn = (front < 8 || back < 8 || left < 8 || right < 8);
  if (warn) {
    updateDisplay(front, back, left, right);
    digitalWrite(LED_WARNING, HIGH);
    tone(BUZZER, 1000); // slow beep
    delay(600);
    digitalWrite(LED_WARNING, LOW);
    noTone(BUZZER);
    delay(600);
    return;
  }
  // ---------------------- NORMAL DISPLAY ----------------------
  updateDisplay(front, back, left, right);
}
