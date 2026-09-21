import time

try:
    import serial
except ImportError:
    serial = None


class SerialComm:
    """
    Sends tracking data to an MCU over UART on every processed frame.

    Uses the Raspberry Pi's primary UART (/dev/serial0) at 115200 baud.
    When pyserial is not installed (e.g. running on a laptop), the
    class silently disables itself so the rest of the pipeline still runs.
    """

    def __init__(self, port="/dev/serial0", baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser = None
        self._open()

    def _open(self):
        if serial is None:
            print("pyserial not installed; serial output disabled.")
            return

        try:
            self._ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout
            )
            print(
                f"Serial: connected to {self.port} @ {self.baudrate} baud"
            )
        except (serial.SerialException, OSError) as error:
            self._ser = None
            print(f"Serial: failed to open {self.port} ({error}); "
                  "serial output disabled.")

    def send(self, rotation, angle, speed):
        """
        Write one line of tracking data to the MCU.

        The payload is sent as raw "data" only:
            rotation,angle,speed\\n

        Args:
            rotation:
                Player rotation in degrees (0 when N/A).
            angle:
                Servo angle in degrees (0 when N/A).
            speed:
                Angular speed in degrees/second (0 when N/A).
        """
        if self._ser is None:
            return

        message = f"{rotation},{angle},{speed}\n"

        try:
            self._ser.write(message.encode("utf-8"))

            if self._ser.in_waiting:
                response = self._ser.readline().decode("utf-8").strip()
                if response:
                    print(f"MCU: {response}")

        except (serial.SerialException, OSError) as error:
            print(f"Serial: write failed ({error}); disabling.")
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None