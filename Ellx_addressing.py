import serial.tools.list_ports
from EllxBus import EllxBus


class EllxAddressing:
    VALID_ADDRS = "0123456789ABCDEF"

    def __init__(self, port: str | None = None):
        self.port = port

    @staticmethod
    def find_ell_ports():
        """
        Find available ELLx-related COM ports by VID/PID.
        Returns a list of port names, e.g. ['COM7', 'COM8'].
        """
        ell_ports = []
        for port in serial.tools.list_ports.comports():
            print(
                f"Found: {port.device} - {port.description} - "
                f"{port.vid}:{port.pid} - {port.serial_number}"
            )
            if port.vid == 1027 and port.pid == 24597:
                ell_ports.append(port.device)
        return ell_ports

    def set_port(self, port: str):
        self.port = port

    def _require_port(self):
        if not self.port:
            raise ValueError("Port is not set. Use set_port(port) or pass port at initialization.")

    def scan_all_addresses(self):
        """
        Scan all possible ELLx bus addresses (0-9, A-F) on the current port.
        Returns a dict: {address: reply}
        """
        self._require_port()
        found = {}

        with EllxBus(self.port) as bus:
            for addr in self.VALID_ADDRS:
                reply = bus.get_info(addr)
                if reply:
                    found[addr] = reply

        return found

    def assign_single_default_device(self, new_addr: str):
        """
        Assumes exactly ONE default-address device ('0') is connected on the bus.
        Changes address 0 -> new_addr and verifies it.
        """
        self._require_port()

        new_addr = new_addr.upper()
        if new_addr not in self.VALID_ADDRS:
            raise ValueError("new_addr must be one hex digit: 0-9 or A-F")

        with EllxBus(self.port) as bus:
            print("[1] Checking default address 0 ...")
            info = bus.get_info("0")
            if not info:
                raise RuntimeError(
                    "No response from address 0. Check connection/power/COM port."
                )
            print("Reply from 0:", info)

            print(f"[2] Changing address 0 -> {new_addr} ...")
            reply = bus.change_address("0", new_addr)
            if not reply:
                raise RuntimeError("No reply after change-address command.")

            print("Change reply:", reply)

            # According to manual, reply should come from new address if accepted.
            addr, code = bus.parse_status(reply)
            if addr != new_addr or code != 0:
                meaning = bus.status_meaning(code) if code is not None else "N/A"
                raise RuntimeError(
                    f"Unexpected change-address reply. "
                    f"addr={addr}, code={code}, meaning={meaning}"
                )

            print(f"[3] Verifying new address {new_addr} ...")
            verify_info = bus.get_info(new_addr)
            if not verify_info:
                raise RuntimeError(
                    f"Changed to {new_addr}, but no response to '{new_addr}in'."
                )

            print("Verified info:", verify_info)
            print("Address assignment completed.")

            return {
                "old_address": "0",
                "new_address": new_addr,
                "change_reply": reply,
                "verified_info": verify_info,
            }