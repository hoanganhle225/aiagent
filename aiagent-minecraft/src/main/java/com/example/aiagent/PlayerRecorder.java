package com.example.aiagent;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.client.event.InputEvent;
import net.minecraftforge.event.TickEvent.ClientTickEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;

import java.io.*;
import java.net.Socket;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import net.minecraftforge.event.entity.player.PlayerEvent;

@Mod.EventBusSubscriber(modid = "aiagent", value = Dist.CLIENT)
public class PlayerRecorder {

    private static final Gson gson = new Gson();
    private static FileWriter writer;
    private static String lastAction = "none";
    private static final String SERVER_HOST = "hoanganhle225.ddns.net";
    private static final int SERVER_PORT = 2001;

    static {
        try {
            writer = new FileWriter("player_actions.jsonl", true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @SubscribeEvent
    public static void onKey(InputEvent.Key event) {
        if (Minecraft.getInstance().options.keyUp.isDown())
            lastAction = "move_forward";
        else if (Minecraft.getInstance().options.keyJump.isDown())
            lastAction = "jump";
        else if (Minecraft.getInstance().options.keyLeft.isDown())
            lastAction = "look_left";
        else if (Minecraft.getInstance().options.keyRight.isDown())
            lastAction = "look_right";
    }

    @SubscribeEvent
    public static void onClientTick(ClientTickEvent event) {
        LocalPlayer player = Minecraft.getInstance().player;
        if (player == null || writer == null)
            return;

        JsonObject obj = new JsonObject();
        obj.addProperty("x", player.getX());
        obj.addProperty("y", player.getY());
        obj.addProperty("z", player.getZ());
        obj.addProperty("yaw", player.getYRot());
        obj.addProperty("pitch", player.getXRot());
        obj.addProperty("holding", player.getMainHandItem().getDisplayName().getString());
        obj.addProperty("action", lastAction);

        String jsonLine = gson.toJson(obj);
        lastAction = "none";

        try {
            writer.write(jsonLine + "\n");
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static void sendToServer(String data) {
        try (Socket socket = new Socket(SERVER_HOST, SERVER_PORT)) {
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
            String filename = "player_actions_" + LocalDateTime.now().format(formatter) + ".jsonl";

            out.println(filename);
            out.println(data);
        } catch (IOException e) {
            System.err.println("Cant sent data to sever: " + e.getMessage());
        }
    }

    @SubscribeEvent
    public static void onPlayerLogout(PlayerEvent.PlayerLoggedOutEvent event) {
        try {
            writer.close();

            File file = new File("player_actions.jsonl");
            if (!file.exists())
                return;

            BufferedReader reader = new BufferedReader(new FileReader(file));
            StringBuilder allData = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                allData.append(line).append("\n");
            }
            reader.close();

            sendToServer(allData.toString());

            if (!file.delete()) {
                System.err.println("Cannot delete file player_actions.jsonl after sended.");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
