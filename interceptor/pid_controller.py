"""PID controller."""


class PIDController:
    """Discrete PID controller. One instance per controlled axis."""

    def __init__(self, kp: float, ki: float, kd: float, output_min: float = -100, output_max: float = 100):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max

        self.error = 0.0
        self.error_last = 0.0
        self.integral_error = 0.0
        self.derivative_error = 0.0
        self.velocity_command = 0.0

    def compute(self, error: float, dt: float) -> float:
        """One PID tick. Returns clamped float velocity command in [output_min, output_max]."""
        if dt <= 0:
            return 0.0

        self.error = error
        self.integral_error += error * dt
        if self.integral_error > self.output_max:
            self.integral_error = self.output_max
        elif self.integral_error < self.output_min:
            self.integral_error = self.output_min
        self.derivative_error = (error - self.error_last) / dt
        self.error_last = error

        output = self.kp * self.error + self.ki * self.integral_error + self.kd * self.derivative_error
        
        if output > self.output_max:
            output = self.output_max
        elif output < self.output_min:
            output = self.output_min

        self.velocity_command = output
        return output

    def reset_integral(self) -> None:
        """
        Zero the integral accumulator. Call when target lost or FSM state changes.
        """
        self.integral_error = 0.0