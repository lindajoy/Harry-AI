import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class WebsocketService {
  private socket: WebSocket | undefined;
  private messageSubject = new Subject<string>();
  public messages$ = this.messageSubject.asObservable(); 

  connect(tone: string): void {
    if (typeof window !== 'undefined') {
      const url = `wss://harry-ai-production.up.railway.app/ws?voice=${encodeURIComponent(tone)}`;
      this.socket = new WebSocket(url);

      this.socket.onopen = () => {
        console.log('🔗 WebSocket connected');
      };

      this.socket.onmessage = (event) => {
        this.messageSubject.next(event.data); 
      };

      this.socket.onclose = () => {
        console.log('❌ WebSocket closed');
      };

      this.socket.onerror = (error) => {
        console.error('🚨 WebSocket error:', error);
      };
    }
  }

  sendMessage(message: string, voice: string): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.close();
    }
  
    const socketUrl = `wss://harry-ai-production.up.railway.app/ws?voice=${encodeURIComponent(voice)}`;
    this.socket = new WebSocket(socketUrl);
    console.log('🔗 WebSocket connected with voice:', voice)
    console.log(socketUrl);
  
    this.socket.onopen = (data) => {
      this.socket!.send(message);
    };

    this.socket.onmessage = (event) => {
      this.messageSubject.next(event.data); 
    };

  
    this.socket.onerror = (err) => {
      console.error('❌ WebSocket error:', err);
    };
  
    this.socket.onclose = () => {
      console.log('🔌 WebSocket closed');
    };
  }

  close(): void {
    this.socket?.close();
  }
}
