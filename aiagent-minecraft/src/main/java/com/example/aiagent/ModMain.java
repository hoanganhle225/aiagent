package com.example.aiagent;

import com.google.gson.Gson;
import com.mojang.blaze3d.platform.InputConstants;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.event.TickEvent.ClientTickEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import org.lwjgl.glfw.GLFW;

import java.io.*;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

@Mod("aiagent")
@Mod.EventBusSubscriber(modid = "aiagent", value = Dist.CLIENT)
public class ModMain {
    private static final Minecraft mc = Minecraft.getInstance();
    private static Socket socket;
    private static PrintWriter out;
    private static BufferedReader in;
    private static boolean shownWelcomeMessage = false;

    public ModMain() {
        MinecraftForge.EVENT_BUS.register(this);
        connectToAIServer();
    }

    private void connectToAIServer() {
        try {
            socket = new Socket("127.0.0.1", 2107);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            System.out.println("[Forge] Connected to AI Server");
        } catch (Exception e) {
            System.err.println("[Forge] Could not connect to AI Server: " + e.getMessage());
        }
    }

    @SubscribeEvent
    public static void onClientTick(ClientTickEvent event) {
        if (mc.player == null || socket == null || socket.isClosed()) {
            return;
        }

        // Hiển thị thông báo chỉ một lần khi đã load xong GUI
        if (!shownWelcomeMessage && mc.gui != null) {
            mc.gui.getChat().addMessage(Component.literal("AI Agent Mod loaded!"));
            shownWelcomeMessage = true;
        }

        try {
            Map<String, Object> state = new HashMap<>();
            state.put("x", mc.player.getX());
            state.put("y", mc.player.getY());
            state.put("z", mc.player.getZ());
            state.put("yaw", mc.player.getYRot());
            state.put("pitch", mc.player.getXRot());
            state.put("holding", mc.player.getMainHandItem().toString());

            out.println(new Gson().toJson(state));

            String response = in.readLine();
            if (response != null) {
                Map<?, ?> actionResponse = new Gson().fromJson(response, Map.class);
                int actionIdx = ((Double) actionResponse.get("action")).intValue();
                performAction(actionIdx);
            }
        } catch (IOException e) {
            System.err.println("[Forge] Communication error: " + e.getMessage());
        }
    }

    private static void performAction(int actionIdx) {
        if (mc.player == null)
            return;

        double speed = 0.1;
        switch (actionIdx) {
            case 1 -> mc.player.setDeltaMovement(0, mc.player.getDeltaMovement().y, speed);
            case 2 -> mc.player.setDeltaMovement(0, mc.player.getDeltaMovement().y, -speed);
            case 3 -> mc.player.setDeltaMovement(-speed, mc.player.getDeltaMovement().y, 0);
            case 4 -> mc.player.setDeltaMovement(speed, mc.player.getDeltaMovement().y, 0);
            default -> mc.player.setDeltaMovement(0, mc.player.getDeltaMovement().y, 0);
        }
    }
}
