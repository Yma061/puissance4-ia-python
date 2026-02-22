import customtkinter as ctk
import numpy as np
from game.board import *
from game.rules import winning_move
from game.ai import minimax
import math

PLAYER_PIECE = 1
AI_PIECE = 2

class Connect4UI(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Puissance 4 - IA")
        self.geometry("700x700")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.board = create_board()
        self.turn = 0  # 0 = joueur, 1 = IA

        self.buttons = []
        self.cells = []

        self.create_widgets()
        self.draw_board()

    def create_widgets(self):

        top_frame = ctk.CTkFrame(self)
        top_frame.pack(pady=10)

        for col in range(COLUMN_COUNT):
            btn = ctk.CTkButton(
                top_frame,
                text=str(col),
                width=60,
                command=lambda c=col: self.player_move(c)
            )
            btn.grid(row=0, column=col, padx=5)
            self.buttons.append(btn)

        self.grid_frame = ctk.CTkFrame(self)
        self.grid_frame.pack(pady=20)

        for r in range(ROW_COUNT):
            row_cells = []
            for c in range(COLUMN_COUNT):
                cell = ctk.CTkLabel(
                    self.grid_frame,
                    text=" ",
                    width=60,
                    height=60,
                    corner_radius=30,
                    fg_color="gray"
                )
                cell.grid(row=r, column=c, padx=5, pady=5)
                row_cells.append(cell)
            self.cells.append(row_cells)

    def draw_board(self):
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT):

                piece = self.board[r][c]

                if piece == PLAYER_PIECE:
                    color = "red"
                elif piece == AI_PIECE:
                    color = "yellow"
                else:
                    color = "gray"

                self.cells[ROW_COUNT - 1 - r][c].configure(fg_color=color)

    def player_move(self, col):

        if is_valid_location(self.board, col):

            row = get_next_open_row(self.board, col)
            drop_piece(self.board, row, col, PLAYER_PIECE)

            if winning_move(self.board, PLAYER_PIECE):
                self.show_winner("Tu as gagné !")
                return

            self.draw_board()
            self.after(300, self.ai_move)

    def ai_move(self):

        col, _ = minimax(self.board, 4, -math.inf, math.inf, True)

        if is_valid_location(self.board, col):

            row = get_next_open_row(self.board, col)
            drop_piece(self.board, row, col, AI_PIECE)

            if winning_move(self.board, AI_PIECE):
                self.show_winner("L'IA a gagné !")
                return

            self.draw_board()

    def show_winner(self, message):
        popup = ctk.CTkToplevel(self)
        popup.geometry("300x150")
        popup.title("Fin de partie")

        label = ctk.CTkLabel(popup, text=message, font=("Arial", 18))
        label.pack(pady=20)

        btn = ctk.CTkButton(popup, text="Quitter", command=self.destroy)
        btn.pack(pady=10)