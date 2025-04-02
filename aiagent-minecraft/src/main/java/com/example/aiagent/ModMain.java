package com.example.aiagent;

import com.google.gson.Gson;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import net.minecraft.world.phys.Vec3;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.  event.TickEvent.ClientTickEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;

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

        // Show welcome message only once
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
                // Map<?, ?> actionMap = new Gson().fromJson(response, Map.class);
                // String action = (String) actionMap.get("move");

                // double yaw = ((Number) actionMap.get("yaw")).doubleValue();
                // double pitch = ((Number) actionMap.get("pitch")).doubleValue();

                // mc.player.setYRot((float) yaw);
                // mc.player.setXRot((float) pitch);

                String action = response.trim();
                System.out.println("[Forge] Received action: " + action);

                // Movement as before
                Vec3 lookVec = mc.player.getLookAngle();
                double speed = 0.1;

                switch (action) {
                    case "move_forward" -> mc.player.setDeltaMovement(lookVec.x * speed, mc.player.getDeltaMovement().y,
                            lookVec.z * speed);
                    case "move_backward" -> mc.player.setDeltaMovement(-lookVec.x * speed,
                            mc.player.getDeltaMovement().y, -lookVec.z * speed);
                    case "move_left" -> mc.player.setDeltaMovement(-lookVec.z * speed, mc.player.getDeltaMovement().y,
                            lookVec.x * speed);
                    case "move_right" -> mc.player.setDeltaMovement(lookVec.z * speed, mc.player.getDeltaMovement().y,
                            -lookVec.x * speed);
                    case "jump" -> mc.player.input.jumping = true;
                    default -> mc.player.setDeltaMovement(0, mc.player.getDeltaMovement().y, 0);
                }
                // Disable all manual input from player
                mc.player.input.forwardImpulse = 0;
                mc.player.input.leftImpulse = 0;
                mc.player.input.jumping = false;
                mc.player.input.shiftKeyDown = false;

                // Lock player's view to prevent manual mouse movement
                mc.player.setYRot(mc.player.getYRot());
                mc.player.setXRot(mc.player.getXRot());

            }

        } catch (IOException e) {
            System.err.println("[Forge] Communication error: " + e.getMessage());
        }
    }

    // private static void performAction(int actionIdx) {
    // // Reset all movement
    // mc.player.input.forwardImpulse = 0;
    // mc.player.input.leftImpulse = 0;

    // switch (actionIdx) {
    // case 1 -> mc.player.input.forwardImpulse = 1.0f; // Move forward
    // case 2 -> mc.player.input.forwardImpulse = -1.0f; // Move backward
    // case 3 -> mc.player.input.leftImpulse = 1.0f; // Move left
    // case 4 -> mc.player.input.leftImpulse = -1.0f; // Move right
    // default -> {
    // } // Stay still
    // }

    // System.out.println("[Forge] performAction() called with action " +
    // actionIdx);
    // }
}