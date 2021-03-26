class MotorControl{
  public:
    byte INA_pin;
    byte INB_pin;
    byte PWM_pin;
  
    byte INA_dir;
    byte INB_dir;
    int PWM_val;

  public: 
    MotorControl(byte INA_pin, byte INB_pin, byte PWM_pin){
      this->INA_pin = INA_pin;
      this->INB_pin = INB_pin;
      this->PWM_pin = PWM_pin;
      pinMode(this->INA_pin, OUTPUT);
      pinMode(this->INB_pin, OUTPUT);
      pinMode(this->PWM_pin, OUTPUT);
    }

   void setPWM(int PWM_val){
      this->PWM_val = PWM_val;
      analogWrite(this->PWM_pin, this->PWM_val);
   }
   
   void forward(){
      digitalWrite(this->INA_pin, HIGH);
      digitalWrite(this->INB_pin, LOW);      
   }

   void backward(){
      digitalWrite(this->INA_pin, LOW);
      digitalWrite(this->INB_pin, HIGH);
   }

   void halt() {
      //analogWrite(this->PWM_pin, 0);
      digitalWrite(this->INA_pin, LOW);
      digitalWrite(this->INB_pin, LOW);
   }
};
