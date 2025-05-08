import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  constructor(private http: HttpClient) {}

  generateResponse(formValue: any) {
    return this.http.post<{message: string}>('http://127.0.0.1:5000/api/ask-question', { formValue });
  }
}
