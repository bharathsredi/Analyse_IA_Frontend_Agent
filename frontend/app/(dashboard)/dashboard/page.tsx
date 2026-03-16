"use client";

import { useEffect, useRef } from "react";
import { useChatStore } from "@/store/chatStore";
import api from "@/lib/api";
import EmptyState from "@/components/chat/EmptyState";
import MessageBubble from "@/components/chat/MessageBubble";
import ChatInput from "@/components/chat/ChatInput";
import MetricsBar from "@/components/charts/MetricsBar";
import ShapChart from "@/components/charts/ShapChart";
import AnomalyTable from "@/components/charts/AnomalyTable";
import { BackendResult } from "@/types";

const MAX_POLLS = 20; // 60 seconds at 3s intervals
const POLL_INTERVAL = 3000;

export default function DashboardPage() {
  const {
    messages,
    language: lang,
    isProcessing,
    currentTaskId,
    addMessage,
    updateMessage,
    setProcessing,
    setCurrentTaskId,
  } = useChatStore();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    };
  }, []);

  const pollTask = async (taskId: string, assistantMsgId: string, attempts = 0) => {
    if (attempts >= MAX_POLLS) {
      updateMessage(assistantMsgId, {
        content:
          lang === "fr"
            ? "Délai d'attente dépassé. L'analyse a pris trop de temps."
            : "Timeout exceeded. The analysis took too long.",
        isStreaming: false,
      });
      setProcessing(false);
      setCurrentTaskId(null);
      return;
    }

    try {
      const { data } = await api.get<BackendResult>(`/agent/status/${taskId}`);

      if (data.status === "SUCCESS") {
        updateMessage(assistantMsgId, {
          content: data.answer || (lang === "fr" ? "Analyse terminée." : "Analysis complete."),
          analysis_result: data.result,
          isStreaming: false,
        });
        setProcessing(false);
        setCurrentTaskId(null);
      } else if (data.status === "FAILURE") {
        updateMessage(assistantMsgId, {
          content:
            lang === "fr"
              ? "Une erreur s'est produite lors de l'analyse des données."
              : "An error occurred while analyzing the data.",
          isStreaming: false,
        });
        setProcessing(false);
        setCurrentTaskId(null);
      } else {
        // PENDING -> keep polling
        pollTimerRef.current = setTimeout(
          () => pollTask(taskId, assistantMsgId, attempts + 1),
          POLL_INTERVAL
        );
      }
    } catch (err) {
      console.error("Task polling error:", err);
      updateMessage(assistantMsgId, {
        content:
          lang === "fr"
            ? "Impossible de récupérer le statut de l'analyse."
            : "Failed to retrieve analysis status.",
        isStreaming: false,
      });
      setProcessing(false);
      setCurrentTaskId(null);
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isProcessing) return;

    // 1. Add User Message
    addMessage({
      role: "user",
      content,
      language: lang,
    });

    // 2. Add empty Assistant Message (Streaming state)
    const assistantMsgId = addMessage({
      role: "assistant",
      content: "",
      language: lang,
      isStreaming: true,
    });

    setProcessing(true);

    // 3. Trigger Backend
    try {
      const { data } = await api.post("/agent/ask", {
        query: content,
        language: lang,
      });

      if (data.task_id) {
        setCurrentTaskId(data.task_id);
        // Start polling logic
        pollTask(data.task_id, assistantMsgId);
      } else {
        throw new Error("No task_id returned");
      }
    } catch (err) {
      console.error("Ask API error:", err);
      updateMessage(assistantMsgId, {
        content:
          lang === "fr"
            ? "Je n'ai pas pu lancer l'analyse en raison d'une erreur de réseau."
            : "I could not start the analysis due to a network error.",
        isStreaming: false,
      });
      setProcessing(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0A1628] relative">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto w-full px-4 py-8">
        <div className="max-w-4xl mx-auto w-full flex flex-col min-h-full">
          {messages.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <EmptyState onSelectPrompt={handleSendMessage} lang={lang} />
            </div>
          ) : (
            <div className="flex flex-col flex-1 pb-4">
              {messages.map((msg) => (
                <div key={msg.id} className="w-full flex flex-col gap-6">
                  <MessageBubble message={msg} />
                  
                  {/* Results Display Area (attached to the assistant message) */}
                  {msg.role === "assistant" && msg.analysis_result && (
                    <div className="w-full pl-12 pr-4 mb-8 space-y-6">
                      {msg.analysis_result.metrics && (
                        <MetricsBar metrics={msg.analysis_result.metrics} />
                      )}
                      
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">
                         {msg.analysis_result.top_features && (
                           <ShapChart features={msg.analysis_result.top_features} />
                         )}
                         {msg.analysis_result.anomalies && (
                           <AnomalyTable anomalies={msg.analysis_result.anomalies} />
                         )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} className="h-4" />
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 w-full z-10">
        <ChatInput 
          onSendMessage={handleSendMessage} 
          isProcessing={isProcessing} 
          lang={lang as 'fr' | 'en'} 
        />
      </div>
    </div>
  );
}
