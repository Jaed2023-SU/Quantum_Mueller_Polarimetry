import time
import serial


class EllxBus:
    """
    Minimal ELLx serial helper for address assignment and query.
    Protocol assumptions:
      - ASCII command strings, e.g. '0in', '0ca1', '1gs'
      - Replies terminated by CR LF
      - 9600-8-N-1
    """

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        # Manual suggests purge/reset behavior on connect in FTDI example.
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(0.05)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def send(self, cmd: str):
        """
        Send ASCII command exactly as protocol string.
        """
        if not isinstance(cmd, str):
            raise TypeError("cmd must be str")
        self.ser.write(cmd.encode("ascii"))
        self.ser.flush()

    def read_line(self) -> str:
        """
        Read one CR/LF terminated line. Returns stripped ASCII.
        """
        raw = self.ser.readline()
        if not raw:
            return ""
        return raw.decode("ascii", errors="replace").strip()

    def query(self, cmd: str, settle_s: float = 0.05, retries: int = 3) -> str:
        """
        Send command and read first non-empty line.
        """
        for _ in range(retries):
            self.ser.reset_input_buffer()
            self.send(cmd)
            time.sleep(settle_s)
            reply = self.read_line()
            if reply:
                return reply
        return ""

    def get_info(self, addr: str) -> str:
        return self.query(f"{addr}in", settle_s=0.1)

    def get_status(self, addr: str) -> str:
        return self.query(f"{addr}gs", settle_s=0.05)

    def change_address(self, old_addr: str, new_addr: str) -> str:
        """
        Send 'AcaX'. Per manual, successful reply should come from NEW address.
        """
        return self.query(f"{old_addr}ca{new_addr}", settle_s=0.1)

    @staticmethod
    def parse_status(reply: str):
        """
        Example: '1GS00' -> ('1', 0)
        """
        if len(reply) < 5 or reply[1:3] != "GS":
            return None, None
        addr = reply[0]
        try:
            code = int(reply[3:5], 16)
        except ValueError:
            return addr, None
        return addr, code

    @staticmethod
    def status_meaning(code: int) -> str:
        table = {
            0: "OK",
            1: "Communication timeout",
            2: "Mechanical timeout",
            3: "Command error or not supported",
            4: "Value out of range",
            5: "Module isolated",
            6: "Module out of isolation",
            7: "Initializing error",
            8: "Thermal error",
            9: "Busy",
            10: "Sensor error",
            11: "Motor error",
            12: "Out of range",
            13: "Over current error",
        }
        return table.get(code, "Reserved/unknown")