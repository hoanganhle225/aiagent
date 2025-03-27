import java.net.ServerSocket;
import java.net.Socket;
import java.io.InputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class SocketTest {
    public static void main(String[] args) throws Exception {
        ServerSocket serverSocket = new ServerSocket(2001);
        System.out.println("🟢 Dang mo cong 2001... Doi client ket noi...");

        Socket socket = serverSocket.accept();
        System.out.println("✅ Client da ket noi!");

        BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));

        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println("📩 Nhan tu client: " + line);
        }

        System.out.println("🔴 Client ngat ket noi. Dong server...");
        reader.close();
        socket.close();
        serverSocket.close();
    }
}
