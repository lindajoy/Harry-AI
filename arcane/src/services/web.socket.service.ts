import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class WebsocketService {
  private socket: WebSocket | undefined;
  private messageSubject = new Subject<string>();
  public messages$ = this.messageSubject.asObservable(); 

  connect(): void {
    if (typeof window !== 'undefined') {
      this.socket = new WebSocket('ws://harry-ai-production.up.railway.app/ws');

      this.socket.onopen = () => {
        console.log('🔗 WebSocket connected');
      };

      this.socket.onmessage = (event) => {
        console.log('📩 Received from server:', event.data);
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

  sendMessage(message: string, voice: boolean): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(message);
    } else {
      console.error('❌ WebSocket not open');
    }
  }

  close(): void {
    this.socket?.close();
  }
}
